"""
Dataset utilities for depth-progressive 3D velocity model building.

Each 3D velocity patch is stored as [nz, ny, nx]. Under the proposed
depth-as-channel formulation, the depth axis nz is later interpreted as the
channel dimension of a 2D U-Net. After batching, tensors therefore have the
NCHW layout [B, nz, ny, nx], where nz acts as the number of channels and
(ny, nx) form the lateral 2D plane.

For each training sample, this dataset returns:
    vp_bottom   : target deep velocity patch, v_deep
    cond_top    : overlapping shallow velocity patch, v_shallow
    struc_bottom: structural attribute for the deep patch, s
    dbottom     : normalized deepest depth coordinate, d_max
    well        : laterally replicated single-well velocity profile, w
    well_loc    : normalized lateral well location, l = (x, y)
    out_dict    : reserved placeholder for compatibility with the training code

Classifier-free guidance dropout is intentionally not implemented here.
The training loop should independently drop `struc_bottom` and `well` so that
the same network learns unconditional, well-only, image-only, and
well-plus-image generation.
"""

import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch
from scipy.signal import convolve
from scipy.io import loadmat
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import OrderedDict
import gc
import glob


def load_data(
    *, data_dir, batch_size, depth_size, device, class_cond=False, deterministic=False
):
    """
    Create an infinite generator of mini-batches for diffusion-model training.

    The underlying BasicDataset returns individual training tuples. PyTorch's
    DataLoader automatically stacks them into batched tensors with dimensions:

        vp_bottom    : [B, nz, ny, nx]
        cond_top     : [B, nz, ny, nx]
        struc_bottom : [B, nz, ny, nx]
        dbottom      : [B, 1]
        well         : [B, nz, ny, nx]
        well_loc     : [B, 2]

    Here, nz = depth_size is treated as the channel dimension by the 2D U-Net.

    Args:
        data_dir: Directory containing precomputed `.npz` samples.
        batch_size: Number of randomly sampled patch pairs per batch.
        depth_size: Number of depth samples in each shallow/deep patch.
        device: Reserved argument for compatibility with the training interface.
                Device placement is handled outside this function.
        class_cond: Reserved class-conditioning flag.
        deterministic: If True, DataLoader shuffling is disabled.

    Yields:
        Batched outputs returned by BasicDataset.__getitem__().
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset = BasicDataset(
        data_dir,
        depth_size,
        class_cond=class_cond,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # Random ordering is preferred because each __getitem__ call already
        # performs random file selection, patch extraction, and augmentation.
        shuffle=not deterministic,
        num_workers=4,
        # Pinning accelerates host-to-GPU transfer in the training loop.
        pin_memory=True,
        prefetch_factor=4,
        # Ensures every optimization step receives a full batch.
        drop_last=True,
    )

    # Use an infinite generator because the dataset is sampled randomly rather
    # than traversed epoch by epoch.
    while True:
        yield from loader


def normalizer_vel(x, dmin=1000, dmax=5000):
    """
    Normalize velocity values from [dmin, dmax] to [-1, 1].

    Diffusion models are trained more stably when the target velocity values
    are placed in a consistent numerical range.

    Args:
        x: Velocity array in physical units, typically m/s.
        dmin: Minimum velocity represented by -1.
        dmax: Maximum velocity represented by +1.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def denormalizer_vel(x, dmin=1000, dmax=5000):
    """
    Convert normalized velocity values from [-1, 1] back to physical units.
    """
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


def normalizer_depth(x, dmin=0, dmax=128):
    """
    Normalize the absolute depth coordinate d_max to [-1, 1].

    In the manuscript, d_max is the depth coordinate of the deepest grid point
    in the target deep patch. It is embedded together with the diffusion
    timestep and injected into residual blocks of the U-Net.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def normalizer_well_loc(x, dmin=0, dmax=255):
    """
    Normalize lateral well coordinates to [-1, 1].

    The normalized well location is later embedded and used to distinguish
    where the replicated well profile originates within the lateral patch.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


class BasicDataset(Dataset):
    """
    Random-access dataset for depth-progressive 3D velocity-model training.

    Each .npz file is expected to contain:
        v3d  : 3D velocity model with shape [nz_total, ny, nx].
        seis : corresponding 3D structural attribute with the same shape.

    The structural attribute is assumed to be precomputed outside this class.
    In the manuscript, it corresponds to a synthetic seismic/convolution image
    derived from vertical reflectivity and a Ricker wavelet.

    A training item is built from two overlapping depth patches:
        v_shallow = [dtop - ds : dtop]
        v_deep    = [dbottom - ds : dbottom]

    where dbottom - dtop = ds / 2. Therefore, the two patches have a fixed
    50% overlap, consistent with the 3D depth-progressive formulation.
    """

    def __init__(
        self,
        paths,
        depth_size,
        class_cond=False,
        cache_size=100,
        preload_size=50,
        num_workers=4,
        file_patterns=None,
    ):
        """
        Args:
            paths: Directory containing `.npz` data files.
            depth_size: Patch depth nz. This later becomes the channel count
                        of the depth-as-channel 2D U-Net.
            class_cond: Reserved flag for compatibility with the training code.
            cache_size: Maximum number of fully loaded samples kept in the
                        least-recently-used cache.
            preload_size: Maximum number of samples held in the asynchronous
                          preload buffer.
            num_workers: Number of background threads for disk preloading.
            file_patterns: Optional glob patterns, such as:
                           ['Overthrust_*.npz', 'SEAMArid_*.npz'].
                           If None, all `.npz` files are included.
        """
        super().__init__()

        self.paths = paths
        self.class_cond = class_cond
        self.ds = depth_size

        # Scan all available precomputed velocity/structure sample files.
        self.file_list = self._scan_files(file_patterns)
        self.num_files = len(self.file_list)

        if self.num_files == 0:
            raise ValueError(f"No npz files found in {paths}")

        print(f"Found {self.num_files} npz files in {paths}")

        # LRU cache reduces repeated disk reads for frequently sampled files.
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()

        # A separate preload buffer stores samples loaded asynchronously before
        # they are requested by __getitem__().
        self.preload_size = min(preload_size, self.num_files)
        self.preload_buffer = {}
        self.preload_lock = threading.Lock()

        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Fill the initial preload buffer before training begins.
        self._preload_initial_batch()

    def _scan_files(self, file_patterns=None):
        """
        Find valid `.npz` sample files in the dataset directory.

        Returns:
            Sorted file names only, without directory prefixes.
        """
        file_list = []

        if not self.paths.endswith("/"):
            self.paths += "/"

        if file_patterns is None:
            file_list = glob.glob(os.path.join(self.paths, "*.npz"))
        else:
            for pattern in file_patterns:
                files = glob.glob(os.path.join(self.paths, pattern))
                file_list.extend(files)

        # Remove duplicates in case multiple patterns overlap.
        file_list = sorted(list(set(file_list)))

        # Store only the base file name because self.paths is added during load.
        file_list = [os.path.basename(f) for f in file_list]

        return file_list

    def _preload_initial_batch(self):
        """
        Randomly preload an initial subset of files in parallel.

        This reduces disk I/O stalls during the first training iterations.
        """
        files_to_preload = random.sample(
            self.file_list,
            min(self.preload_size, self.num_files),
        )

        futures = []

        for filename in files_to_preload:
            future = self.executor.submit(self._load_npz, filename)
            futures.append((filename, future))

        for filename, future in futures:
            try:
                data = future.result()

                with self.preload_lock:
                    self.preload_buffer[filename] = data

            except Exception as e:
                print(f"Error preloading {filename}: {e}")

    def _load_npz(self, filename):
        """
        Load one precomputed 3D velocity sample and its structural attribute.

        Required `.npz` keys:
            v3d : 3D velocity model.
            seis: Structural attribute aligned with v3d.

        Returns:
            Dictionary containing velocity, structural attribute, and filename.
        """
        filepath = os.path.join(self.paths, filename)

        try:
            data = np.load(filepath)

            if "v3d" not in data or "seis" not in data:
                raise KeyError(
                    f"Required keys 'v3d' or 'seis' not found in {filename}"
                )

            return {
                "v3d": data["v3d"],
                # `ref` denotes the seismic-derived structural constraint.
                "ref": data["seis"],
                "filename": filename,
            }

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise

    def _get_from_cache(self, filename):
        """
        Retrieve one sample from the LRU cache, preload buffer, or disk.

        Cache priority:
            1. Existing LRU cache entry.
            2. Background preload buffer.
            3. Direct synchronous file loading.

        The accessed item is inserted into the LRU cache and the least recently
        used item is removed when the configured cache capacity is exceeded.
        """
        with self.cache_lock:
            if filename in self.cache:
                self.cache.move_to_end(filename)
                return self.cache[filename]

            with self.preload_lock:
                if filename in self.preload_buffer:
                    data = self.preload_buffer.pop(filename)
                else:
                    data = None

            if data is None:
                data = self._load_npz(filename)

            self.cache[filename] = data

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

            return data

    def _async_preload(self, filename):
        """
        Schedule background loading of a file that is not already cached.

        The actual loading is performed by `_preload_single()` in a thread.
        """
        with self.cache_lock:
            in_cache = filename in self.cache

        with self.preload_lock:
            in_preload = filename in self.preload_buffer

        if not in_cache and not in_preload:
            self.executor.submit(self._preload_single, filename)

    def _preload_single(self, filename):
        """
        Load one file asynchronously and place it in the preload buffer.

        The buffer size is capped to avoid holding too many large 3D volumes
        in CPU memory.
        """
        try:
            data = self._load_npz(filename)

            with self.preload_lock:
                if len(self.preload_buffer) < self.preload_size:
                    self.preload_buffer[filename] = data

        except Exception as e:
            print(f"Error in async preload of {filename}: {e}")

    def __len__(self):
        """
        Return a large virtual length.

        Samples are randomly generated on demand, so the dataset does not have
        a conventional finite epoch length.
        """
        return 1000000

    def __getitem__(self, idx):
        """
        Construct one randomized depth-progressive training sample.

        Returns:
            vp_bottom:
                Normalized target deep velocity patch v_deep.
                Shape: [ds, ny, nx].

            cond_top:
                Normalized shallow conditioning patch v_shallow.
                Shape: [ds, ny, nx].

            struc_bottom:
                Structural attribute aligned with the target patch.
                Shape: [ds, ny, nx].

            dbottom:
                Normalized scalar d_max, namely the absolute coordinate of the
                deepest sample in v_deep.
                Shape: [1].

            well:
                Dense representation of one well-log velocity profile. The
                selected 1D well trace is repeated over the lateral plane.
                Shape: [ds, ny, nx].

            well_loc:
                Normalized lateral coordinates of the selected well.
                Shape: [2].

            out_dict:
                Empty placeholder reserved for compatibility with the existing
                diffusion-model training interface.
        """
        # Randomly sample a model with replacement. This lets every batch mix
        # samples from different geological models and augmentations.
        filename = random.choice(self.file_list)

        # Proactively preload several future candidate files to hide disk I/O.
        next_files = random.sample(
            self.file_list,
            min(3, self.num_files),
        )

        for next_file in next_files:
            self._async_preload(next_file)

        data = self._get_from_cache(filename)

        # Expected volume layout: [depth, lateral_y, lateral_x].
        vp_mm = data["v3d"]
        ref_mm = data["ref"]

        # ---------------------------------------------------------------
        # Rotation augmentation in the lateral plane.
        #
        # Rotations are applied around the vertical/depth axis, preserving
        # depth ordering while increasing lateral structural diversity.
        # The four possibilities are 0°, 90°, 180°, and 270°.
        # ---------------------------------------------------------------
        random_val = random.uniform(0, 1)

        if 0.5 > random_val >= 0.25:
            vp_mm = np.rot90(vp_mm, k=1, axes=(1, 2))
            ref_mm = np.rot90(ref_mm, k=1, axes=(1, 2))

        elif 0.75 > random_val >= 0.5:
            vp_mm = np.rot90(vp_mm, k=2, axes=(1, 2))
            ref_mm = np.rot90(ref_mm, k=2, axes=(1, 2))

        elif random_val >= 0.75:
            vp_mm = np.rot90(vp_mm, k=3, axes=(1, 2))
            ref_mm = np.rot90(ref_mm, k=3, axes=(1, 2))

        nz, ny, nx = vp_mm.shape

        # ---------------------------------------------------------------
        # Construct a shallow/deep patch pair with fixed 50% overlap.
        #
        # ds   : patch depth, which becomes the 2D U-Net channel count.
        # dgap : offset between shallow and deep patch ends.
        #
        # Since dgap = ds / 2, the two depth patches overlap by ds / 2.
        # This fixed overlap makes d_max sufficient to identify the absolute
        # depth position of each target patch, as described in the manuscript.
        # ---------------------------------------------------------------
        dgap = self.ds // 2

        # dbottom is the exclusive end index of the deep target patch.
        # Its valid range ensures both shallow and deep patches remain inside
        # the full 3D velocity volume.
        dbottom = random.randint(self.ds + dgap, nz)

        # dtop is the exclusive end index of the shallow conditioning patch.
        dtop = dbottom - dgap

        # Shallow patch: v_shallow.
        vp_top = vp_mm[dtop - self.ds:dtop]

        # Deep target patch: v_deep.
        vp_bottom = vp_mm[dbottom - self.ds:dbottom]

        # Structural constraint aligned with the target deep patch.
        ref_bottom = ref_mm[dbottom - self.ds:dbottom]

        # Normalize only the velocity variables. The seismic/structural
        # attribute is assumed to be pre-scaled during data preparation.
        vp_top = normalizer_vel(vp_top)
        vp_bottom = normalizer_vel(vp_bottom)

        # ---------------------------------------------------------------
        # Simulate a single well constraint.
        #
        # A lateral location is selected randomly, and its full-depth velocity
        # trace within the target patch is extracted. The 1D well trace is then
        # replicated over the lateral plane, while `well_loc` preserves the
        # true well position for the network's location embedding.
        # ---------------------------------------------------------------
        well_locx = random.randint(0, ny - 1)
        well_locy = random.randint(0, nx - 1)

        # Keep the original indexing convention used by the existing code.
        # For non-square lateral grids, verify that coordinate naming and
        # indexing remain consistent with the intended [z, y, x] convention.
        well = vp_bottom[:, well_locy, well_locx]

        # Expand the 1D well profile to the same shape as v_deep:
        # [ds] -> [ds, ny, nx].
        well = np.tile(
            well[:, np.newaxis, np.newaxis],
            (1, ny, nx),
        )

        # d_max is the deepest coordinate of the target patch. It is passed as
        # a scalar condition rather than as a spatial depth-coordinate channel.
        dbottom = np.array([dbottom], dtype=np.float32)
        dbottom = normalizer_depth(dbottom)

        # Lateral well coordinate, later embedded by the network.
        well_loc = np.array([well_locy, well_locx], dtype=np.float32)
        well_loc = normalizer_well_loc(well_loc, dmax=nx)

        # Explicit float32 conversion avoids unnecessary float64 tensors after
        # PyTorch DataLoader collation.
        cond_top = np.array(vp_top, dtype=np.float32)
        struc_bottom = np.array(ref_bottom, dtype=np.float32)
        well = np.array(well, dtype=np.float32)

        # Placeholder retained for compatibility with the existing diffusion
        # training pipeline. Conditional dropout should be applied later:
        #
        #   - set struc_bottom to None with probability p
        #   - set well (and well_loc) to None with probability p
        #
        # The two operations should be independent.
        out_dict = {}

        return (
            vp_bottom,
            cond_top,
            struc_bottom,
            dbottom,
            well,
            well_loc,
            out_dict,
        )

    def get_file_statistics(self):
        """
        Count samples grouped by the filename prefix.

        This is useful for checking the relative contribution of different
        source models, for example Overthrust, SEAM Arid, or SEG/EAGE.
        """
        stats = {}

        for filename in self.file_list:
            prefix = filename.split("_")[0] if "_" in filename else "unknown"
            stats[prefix] = stats.get(prefix, 0) + 1

        return stats

    def __del__(self):
        """
        Shut down the background preload executor when the dataset is released.
        """
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)