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
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
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
        shuffle=not deterministic, 
        num_workers=4, 
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True
    )

    while True:
        yield from loader

def normalizer_vel(x, dmin=1000, dmax=5000):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def denormalizer_vel(x, dmin=1000, dmax=5000):
    return 0.5 * (x + 1) * (dmax - dmin) + dmin

def normalizer_depth(x, dmin=0, dmax=128):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_well_loc(x, dmin=0, dmax=255):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

class BasicDataset(Dataset):
    def __init__(self, paths, depth_size, class_cond=False, 
                 cache_size=100, preload_size=50, num_workers=4,
                 file_patterns=None):
        """
        Args:
            paths: 数据集目录路径
            depth_size: 深度尺寸
            dt: 时间步长
            class_cond: 类条件标志
            cache_size: 缓存大小
            preload_size: 预加载大小
            num_workers: 工作线程数
            file_patterns: 文件名模式列表，如 ['Overthrust_*.npz', 'SEAMArid_*.npz']
                          如果为None，则扫描所有.npz文件
        """
        super().__init__()
        self.paths = paths
        self.class_cond = class_cond
        self.ds = depth_size
        
        self.file_list = self._scan_files(file_patterns)
        self.num_files = len(self.file_list)
        
        if self.num_files == 0:
            raise ValueError(f"No npz files found in {paths}")
        
        print(f"Found {self.num_files} npz files in {paths}")
        
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        
        self.preload_size = min(preload_size, self.num_files)
        self.preload_buffer = {}
        self.preload_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # pre load
        self._preload_initial_batch()
    
    def _scan_files(self, file_patterns=None):
        file_list = []
        
        if not self.paths.endswith('/'):
            self.paths += '/'
        
        if file_patterns is None:
            file_list = glob.glob(os.path.join(self.paths, '*.npz'))
        else:
            for pattern in file_patterns:
                files = glob.glob(os.path.join(self.paths, pattern))
                file_list.extend(files)
        
        file_list = sorted(list(set(file_list)))
        
        file_list = [os.path.basename(f) for f in file_list]
        
        return file_list
    
    def _preload_initial_batch(self):
        files_to_preload = random.sample(self.file_list, 
                                       min(self.preload_size, self.num_files))
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
        filepath = os.path.join(self.paths, filename)
        try:
            data = np.load(filepath)
            if 'v3d' not in data or 'seis' not in data:
                raise KeyError(f"Required keys 'v3d' or 'seis' not found in {filename}")
            
            return {
                'v3d': data['v3d'],
                'ref': data['seis'],
                'filename': filename  
            }
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            raise
    
    def _get_from_cache(self, filename):
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
        with self.cache_lock:
            in_cache = filename in self.cache
        
        with self.preload_lock:
            in_preload = filename in self.preload_buffer
        
        if not in_cache and not in_preload:
            self.executor.submit(self._preload_single, filename)
    
    def _preload_single(self, filename):
        try:
            data = self._load_npz(filename)
            with self.preload_lock:
                if len(self.preload_buffer) < self.preload_size:
                    self.preload_buffer[filename] = data
        except Exception as e:
            print(f"Error in async preload of {filename}: {e}")
    
    def __len__(self):
        return 1000000
    
    def __getitem__(self, idx):
        filename = random.choice(self.file_list)
        
        next_files = random.sample(self.file_list, min(3, self.num_files))
        for next_file in next_files:
            self._async_preload(next_file)
        
        data = self._get_from_cache(filename)
        vp_mm = data['v3d']
        ref_mm = data['ref']
        
        # roation to augment
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
        dgap = self.ds // 2
        dbottom = random.randint(self.ds + dgap, nz)
        dtop = dbottom - dgap
        
        vp_top = vp_mm[dtop - self.ds:dtop]
        vp_bottom = vp_mm[dbottom - self.ds:dbottom]
        ref_bottom = ref_mm[dbottom - self.ds:dbottom]
        
        vp_top = normalizer_vel(vp_top)
        vp_bottom = normalizer_vel(vp_bottom)
        
        well_locx = random.randint(0, ny - 1)
        well_locy = random.randint(0, nx - 1)
        well = vp_bottom[:, well_locy, well_locx]
        well = np.tile(well[:, np.newaxis, np.newaxis], (1, ny, nx))
        
        dbottom = np.array([dbottom], dtype=np.float32)
        dbottom = normalizer_depth(dbottom)
        
        well_loc = np.array([well_locy, well_locx], dtype=np.float32)
        well_loc = normalizer_well_loc(well_loc, dmax=nx)
        
        cond_top = np.array(vp_top, dtype=np.float32)
        struc_bottom = np.array(ref_bottom, dtype=np.float32)
        well = np.array(well, dtype=np.float32)
        
        out_dict = {}

        return vp_bottom, cond_top, struc_bottom, dbottom, well, well_loc, out_dict
    
    def get_file_statistics(self):

        stats = {}
        for filename in self.file_list:
            prefix = filename.split('_')[0] if '_' in filename else 'unknown'
            stats[prefix] = stats.get(prefix, 0) + 1
        return stats
    
    def __del__(self):

        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
