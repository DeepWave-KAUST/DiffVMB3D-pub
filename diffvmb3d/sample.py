"""
Inference (sampling) script for DiffVMB3D: Depth-progressive 3D velocity model
building via 2D generative diffusion models.

This script implements the complete depth-progressive recursive inference
pipeline described in Section III.II (Algorithm 1) and the depth-attenuated
Gaussian blending scheme from Section III.IV for assembling overlapping patches
into a seamless full-volume velocity prediction.

Inference pipeline overview:
  1. Load the trained U-Net f_theta (EMA weights) and configure the diffusion
     process with timestep respacing (default: 10-step DDIM).
  2. For each test velocity model, tile the 3D volume into overlapping patches
     along the depth (z), inline (y), and crossline (x) directions.
  3. For each lateral column (y, x):
     a. Start from the known shallow velocity patch v_shallow as the initial
        condition (depth level 0).
     b. Recursively generate deeper patches via DDIM sampling, where each
        prediction is conditioned on the previously predicted (shallower) patch.
     c. Optionally inject well-log (w, l) and structural attribute (s) constraints.
  4. Merge overlapping patches using depth-attenuated Gaussian blending:
     - In depth (z): Gaussian weighting with sigma_z = nz/4, combined with a
       linear depth-decay factor that attenuates deeper predictions (Eq. 5-6).
     - In lateral (x, y): 2D Gaussian weighting centered on each patch to
       produce smooth transitions at patch boundaries.
  5. Compute the ensemble mean and standard deviation from batch_size independent
     samples (default 50) as the final prediction and uncertainty estimate.

Usage:
    python sample.py --model_path ./checkpoints/trained_model.pt --use_ddim True
                     --timestep_respacing ddim10 --batch_size 50
                     --use_well True --use_ref True
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from code.datasets import (
    normalizer_vel,
    denormalizer_vel,
    normalizer_depth,
    normalizer_well_loc,
)
import scipy.io as sio
from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import time


def main(batch_size, use_well, use_ref):
    """
    Run depth-progressive inference for all test velocity models.

    This function iterates over a list of test models, applying the full
    depth-progressive sampling pipeline (Algorithm 1) to each one, and
    saves the ensemble mean/std predictions to .mat files.

    Args:
        batch_size:  Number of independent samples to generate per patch
                     for ensemble statistics (50 in the paper).
        use_well:    If True, condition on well-log velocity (w, l).
        use_ref:     If True, condition on structural attribute (s).
    """
    args = create_argparser().parse_args()

    # Override CLI defaults with function arguments for batch experimentation.
    args.batch_size = batch_size
    args.use_well = use_well
    args.use_ref = use_ref

    device = th.device('cuda')

    logger.configure()

    # Output directory organized by sampling method (DDPM vs DDIM with step count).
    if not args.use_ddim:
        dir_output = f'./output/ddpm/'
    else:
        dir_output = f'./output/{args.timestep_respacing}/'
    os.makedirs(dir_output, exist_ok=True)

    # === Step 1: Create model and diffusion with timestep respacing ===
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    # For inference, timestep_respacing="ddim10" creates a SpacedDiffusion
    # with only 10 reverse steps instead of the full T=1000.
    model, diffusion = create_model_and_diffusion(
        **params,
        use_wellguide=args.use_wellguide,
    )

    # Load trained EMA weights (more stable than raw training weights).
    model.load_state_dict(
        th.load(f'{args.model_path}', map_location=device)
    )
    model.to(device=device)
    model.eval()

    # Select the sampling function: DDPM (full chain) or DDIM (accelerated).
    # DDIM with 10 steps is the default for DiffVMB3D inference (Section IV).
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    criterion = th.nn.MSELoss()

    depth_size = args.out_channels  # nz: depth samples per patch (16)

    # === Precompute Gaussian blending weights ===

    # Depth-direction Gaussian weight (Section III.IV, Eq. 5):
    # A 1D Gaussian centered at the middle of the patch along z, with
    # sigma_z = nz/4. This assigns higher weight to the center of each
    # depth patch and lower weight to the overlap regions at top/bottom.
    sigma_z = depth_size // 4
    z = np.arange(depth_size) - depth_size // 2
    gaussian_z = np.exp(-(z ** 2) / (2 * sigma_z ** 2))
    gaussian_z = th.tensor(gaussian_z, dtype=th.float32).to(device).view(depth_size, 1, 1)
    gaussian_z = gaussian_z.repeat(1, args.width_size, args.width_size)

    # Lateral (x-y) Gaussian weight (Section III.IV):
    # A 2D Gaussian centered on the patch in the lateral plane, with
    # sigma_xy = width_size/8. This ensures smooth lateral blending where
    # adjacent patches overlap.
    sigma_xy = args.width_size // 8
    x = np.arange(args.width_size) - args.width_size // 2
    y = np.arange(args.width_size) - args.width_size // 2
    x, y = np.meshgrid(x, y)
    gaussian_xy_ori = gaussian_weight(th.from_numpy(x).float(), th.from_numpy(y).float(), sigma_xy)
    gaussian_xy_ori = gaussian_xy_ori.to(device=device).unsqueeze(0)

    # Step sizes for patch tiling with overlap:
    # Depth overlap = nz/2, lateral overlap = width_size/8.
    step_size_z = depth_size - depth_size // 2               # 50% depth overlap
    step_size_xy = args.width_size - args.width_size // 8    # 87.5% lateral overlap

    # === Step 2: Iterate over test velocity models ===
    md_list = ['Overthrust', 'SEGEAGE', 'SEAMArid', 'Marmousi']
    for md in md_list:
        start = time.time()
        print(f'Sampling start for {md} with batch size {args.batch_size} usewell {args.use_well} useref {args.use_ref}')

        # Load the ground-truth 3D velocity model and structural attribute.
        dict = sio.loadmat(f'../dataset/test/{md}.mat')
        vp = dict['v3d']       # 3D velocity model [nz, ny, nx]
        ref = dict['seis']     # Structural attribute [nz, ny, nx] (e.g., convolution image)

        # Normalize velocity to [-1, 1] for the diffusion model.
        vp = normalizer_vel(vp)
        nz, ny, nx = vp.shape

        vp = th.tensor(vp, dtype=th.float32).unsqueeze(0).to(device=device)    # [1, nz, ny, nx]
        struc = th.tensor(ref, dtype=th.float32).unsqueeze(0).to(device=device) # [1, nz, ny, nx]

        # === Step 3: Compute patch tiling indices ===
        # Generate starting indices for depth, inline, and crossline patches.
        # Ensure the last patch covers the volume boundary (bottom/right edge).

        # Depth indices: start from step_size_z (first patch is the shallow prior),
        # stride by step_size_z, and ensure the deepest patch reaches nz.
        indices_z = list(range(step_size_z, nz - depth_size + 1, step_size_z))
        if indices_z[-1] != nz - depth_size:
            indices_z.append(nz - depth_size)

        # Inline (y) indices
        indices_y = list(range(0, ny - args.width_size + 1, step_size_xy))
        if indices_y[-1] != ny - args.width_size:
            indices_y.append(ny - args.width_size)

        # Crossline (x) indices
        indices_x = list(range(0, nx - args.width_size + 1, step_size_xy))
        if indices_x[-1] != nx - args.width_size:
            indices_x.append(nx - args.width_size)

        # === Step 4: Compute local well coordinates within each lateral patch ===
        # For each patch, determine whether the global well position falls inside.
        # If yes, compute the local (patch-relative) well index; otherwise mark as -1.
        local_well_indicesx = [
            (args.global_well_x - ix) if ix <= args.global_well_x < ix + args.width_size else -1
            for ix in indices_x
        ]
        local_well_indicesy = [
            (args.global_well_y - iy) if iy <= args.global_well_y < iy + args.width_size else -1
            for iy in indices_y
        ]

        print(f'local_well_indicesx: {local_well_indicesx}')
        print(f'local_well_indicesy: {local_well_indicesy}')

        # === Step 5: Compute depth-attenuated patch weights (Section III.IV, Eq. 6) ===
        # Deeper patches receive lower blending weights to account for the
        # recursive error propagation inherent in the depth-progressive scheme:
        # patches generated later (deeper) are conditioned on earlier predictions
        # that may already contain errors.
        # The weight decays linearly from 1.0 (shallowest) to min_weight (deepest).
        min_weight = 0.1
        num_patches = len(indices_z)
        patch_weights = {}
        for patch_idx, iz in enumerate(indices_z):
            if num_patches == 1:
                patch_weights[iz] = 1.0
            else:
                # Linear interpolation: w = 1.0 at patch_idx=0, w = min_weight at patch_idx=num_patches-1
                patch_weights[iz] = 1.0 - (1.0 - min_weight) * patch_idx / (num_patches - 1)

        print(patch_weights)

        # === Step 6: Allocate accumulators for the full-volume prediction ===
        # Weighted accumulators for Gaussian blending across all patches.
        total_pred_mean = th.zeros((nz, ny, nx), dtype=th.float32, device=device)
        total_pred_std = th.zeros((nz, ny, nx), dtype=th.float32, device=device)
        total_weight = th.zeros_like(total_pred_mean)

        # Expand lateral Gaussian weight to full depth for volume-level blending.
        gaussian_xy = gaussian_xy_ori.repeat(nz, 1, 1)

        # === Step 7: Depth-progressive recursive inference (Algorithm 1) ===
        # Outer loops: tile across the lateral plane (y, x).
        # Inner loop: recurse from shallow to deep along z.
        for idy, iy in enumerate(indices_y):
            for idx, ix in enumerate(indices_x):
                # Per-column accumulators for depth-direction blending.
                accumulated_mean = th.zeros((nz, args.width_size, args.width_size), dtype=th.float32).to(device=device)
                accumulated_std = th.zeros((nz, args.width_size, args.width_size), dtype=th.float32).to(device=device)
                accumulated_weight = th.zeros((nz, args.width_size, args.width_size), dtype=th.float32).to(device=device)

                # Initialize with the known shallow velocity prior v_shallow
                # (the top nz depth samples, assumed known from shallow processing).
                vp_top = vp[:, :depth_size, iy:iy + args.width_size, ix:ix + args.width_size]

                # Accumulate the shallow prior with full confidence (weight = 1.0).
                accumulated_mean[:depth_size] += 1.0 * gaussian_z * denormalizer_vel(vp_top[0])
                accumulated_weight[:depth_size] += 1.0 * gaussian_z

                # Replicate for batch sampling (batch_size independent realizations).
                vp_top = vp_top.repeat(args.batch_size, 1, 1, 1)

                # Precompute normalized well location for this lateral patch
                # (only if the well falls within this patch).
                lwx = local_well_indicesx[idx]
                lwy = local_well_indicesy[idy]
                if lwx >= 0 and lwy >= 0:
                    wl_np = np.array([lwy, lwx], dtype=np.float32)
                    wl_norm = normalizer_well_loc(wl_np, dmax=args.width_size)
                    wlt = th.tensor(wl_norm, dtype=th.float32, device=device).repeat(args.batch_size, 1)

                # --- Inner loop: depth-progressive recursion ---
                # At each depth level iz, generate v_deep conditioned on the
                # previous v_shallow (= vp_top from the last iteration).
                for iz in indices_z:
                    print(f'sampling depth grid index_z {iz}, index_y {iy}, index_x {ix}')

                    # Extract the ground-truth deep patch (for well conditioning only).
                    vp_bottom = vp[:, iz:iz + depth_size, iy:iy + args.width_size, ix:ix + args.width_size]

                    # Structural attribute conditioning: extract the corresponding
                    # depth patch, or set to None/zeros if not using structural constraint.
                    if args.use_ref:
                        struc_bottom = struc[:, iz:iz + depth_size, iy:iy + args.width_size, ix:ix + args.width_size]
                    else:
                        struc_bottom = th.zeros_like(vp_bottom)

                    # Depth position embedding: d_max = depth_size + iz (deepest
                    # grid index of the current patch), normalized to [-1, 1].
                    depth_bottom = th.tensor(depth_size + iz, dtype=th.float32, device=device).view(1, 1)
                    depth_bottom = normalizer_depth(depth_bottom)

                    # Replicate conditioning tensors for batch sampling.
                    struc_bottom = struc_bottom.repeat(args.batch_size, 1, 1, 1)
                    depth_bottom = depth_bottom.repeat(args.batch_size, 1)

                    # Well-log conditioning: extract the well velocity profile at
                    # the well location, expand to patch dimensions, and create
                    # the spatial mask indicating the well position.
                    if args.use_well and lwx >= 0 and lwy >= 0:
                        use_wellguide = args.use_wellguide
                        # Extract the 1D well velocity and expand laterally.
                        well_bottom = vp_bottom[:, :, lwy, lwx].unsqueeze(2).unsqueeze(3)
                        well_bottom = well_bottom.repeat(args.batch_size, 1, args.width_size, args.width_size)
                        # Binary mask: 1 at the well location, 0 elsewhere.
                        mask = th.zeros_like(well_bottom)
                        mask[:, :, lwy, lwx] = 1
                    else:
                        # No well data available for this patch.
                        use_wellguide = False
                        well_bottom = None
                        wlt = None

                    # Set structural attribute to None for classifier-free unconditional path.
                    if not args.use_ref:
                        struc_bottom = None

                    # === Run DDIM (or DDPM) sampling ===
                    # Generate batch_size independent velocity samples for the
                    # current depth patch, conditioned on:
                    #   - vp_top:        shallow velocity from previous recursion
                    #   - struc_bottom:  structural attribute s (or None)
                    #   - depth_bottom:  depth position d_max
                    #   - well_bottom:   well velocity w (or None)
                    #   - wlt:           well position l (or None)
                    #   - mask:          well spatial mask (for gradient guidance)
                    sample, _, _, loss_before, loss_after = sample_fn(
                        model, vp_top, struc_bottom, depth_bottom, well_bottom, wlt, mask if args.use_well else None,
                        (args.batch_size, args.out_channels, args.width_size, args.width_size),
                        scale_factor=args.scale_factor if use_wellguide else None,
                        clip_denoised=args.clip_denoised,
                    )

                    # Update the shallow condition for the next depth level:
                    # the current prediction becomes the v_shallow for the next
                    # deeper patch (depth-progressive recursion, Section III.II).
                    vp_top = sample.clone()

                    # === Depth-attenuated Gaussian blending (Section III.IV) ===
                    # Combine the depth-direction Gaussian weight (gaussian_z)
                    # with the linear depth-decay factor (patch_weight) to produce
                    # the final blending weight for this patch.
                    patch_weight = patch_weights[iz]

                    # Accumulate ensemble mean and std with depth-attenuated weights.
                    accumulated_mean[iz:iz + depth_size] += patch_weight * gaussian_z * denormalizer_vel(sample).mean(dim=0)
                    accumulated_std[iz:iz + depth_size] += patch_weight * gaussian_z * denormalizer_vel(sample).std(dim=0)
                    accumulated_weight[iz:iz + depth_size] += patch_weight * gaussian_z

                # Normalize the depth-blended column by accumulated weights.
                final_pred_mean = accumulated_mean / accumulated_weight
                final_pred_std = accumulated_std / accumulated_weight

                # === Lateral Gaussian blending ===
                # Blend this column into the full-volume prediction using the
                # 2D lateral Gaussian weight centered on this patch.
                total_pred_mean[:, iy:iy + args.width_size, ix:ix + args.width_size] += gaussian_xy * final_pred_mean
                total_pred_std[:, iy:iy + args.width_size, ix:ix + args.width_size] += gaussian_xy * final_pred_std
                total_weight[:, iy:iy + args.width_size, ix:ix + args.width_size] += gaussian_xy

        # Normalize the full volume by the total accumulated lateral weights.
        total_pred_mean /= total_weight
        total_pred_std /= total_weight

        # Compute the MSE between the ensemble mean prediction and the ground truth.
        with th.no_grad():
            accs = criterion(total_pred_mean, denormalizer_vel(vp.squeeze()))

        # === Save results ===
        if use_wellguide:
            file_name = f'{dir_output}{md}_batch{args.batch_size}_usewell{args.use_well}_useref{args.use_ref}_wellguide{use_wellguide}_scale{args.scale_factor}_out.mat'
        else:
            file_name = f'{dir_output}{md}_batch{args.batch_size}_usewell{args.use_well}_useref{args.use_ref}_wellguide{use_wellguide}_out.mat'

        sio.savemat(file_name,
                    {'pred_mean': total_pred_mean.squeeze().cpu().numpy(),   # Ensemble mean velocity
                     'pred_std': total_pred_std.squeeze().cpu().numpy(),     # Ensemble std (uncertainty)
                     'accs': accs.item(),                                    # MSE vs ground truth
                     'loss_before': np.array(loss_before, dtype=np.float32), # Well loss before guidance
                     'loss_after': np.array(loss_after, dtype=np.float32)})  # Well loss after guidance

        end = time.time()
        print(f'model {md} inference time cost {end - start} s')

    logger.log("sampling complete")


def gaussian_weight(x, y, sigma):
    """
    Compute a 2D Gaussian weight map for lateral patch blending.

    Used to produce smooth transitions between overlapping patches in the
    inline (y) and crossline (x) directions (Section III.IV).

    Args:
        x:      2D meshgrid of x-coordinates (centered at 0).
        y:      2D meshgrid of y-coordinates (centered at 0).
        sigma:  Standard deviation of the Gaussian.

    Returns:
        2D tensor of Gaussian weights, shape matching x and y.
    """
    return th.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2))).float()


def gaussian_1d(length, sigma):
    """
    Compute a 1D Gaussian weight vector, centered at the midpoint.

    Args:
        length:  Length of the output vector.
        sigma:   Standard deviation of the Gaussian.

    Returns:
        1D tensor of Gaussian weights of the given length.
    """
    coord = th.arange(length).float() - (length - 1) / 2
    return th.exp(-coord ** 2 / (2 * sigma ** 2))


def create_argparser():
    """
    Build the argument parser for inference-specific hyperparameters.

    Inference-specific defaults:
        clip_denoised:       Clamp predicted x_0 to [-1, 1] during sampling.
        use_ddim:            If True, use DDIM sampling (default); otherwise DDPM.
        batch_size:          Number of independent samples per patch for ensemble
                             statistics (50 in the paper).
        model_path:          Path to the trained model checkpoint (EMA weights).
        width_size:          Lateral patch size in grid points (ny = nx per patch).
        dt:                  (Unused) placeholder for continuous-time formulation.
        use_well:            If True, condition on well-log velocity.
        use_ref:             If True, condition on structural attribute.
        use_wellguide:       If True, apply well-log gradient guidance during sampling.
        scale_factor:        Gradient guidance step size for well constraint.
        global_well_x:       Global x-index of the well in the full 3D volume.
        global_well_y:       Global y-index of the well in the full 3D volume.

    Returns:
        argparse.ArgumentParser with all registered arguments.
    """
    defaults = dict(
        clip_denoised=True,       # Clamp x_0 predictions to [-1, 1]
        use_ddim=True,            # Use DDIM sampling (10-step default)
        batch_size=50,            # Ensemble size for mean/std estimation
        model_path="../checkpoints/trained_model.pt",  # Trained EMA checkpoint
        width_size=128,           # Lateral patch size (ny = nx)
        dt=1e-3,                  # Placeholder (unused)
        use_well=False,           # Well-log conditioning flag
        use_ref=False,            # Structural attribute conditioning flag
        use_wellguide=False,      # Well gradient guidance flag
        scale_factor=20,          # Guidance gradient step size
        global_well_x=50,        # Global well x-position in the full volume
        global_well_y=60,        # Global well y-position in the full volume
    )
    # Merge with model/diffusion defaults for full CLI configurability.
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Exhaustive evaluation over all conditioning combinations:
    # batch_size in {1, 50}, use_well in {True, False}, use_ref in {True, False}
    # This produces results for all four conditioning scenarios described in
    # the paper: unconditional, well-only, image-only, and well+image.
    batch_list = [1, 50]
    use_well_list = [True, False]
    use_ref_list = [True, False]
    for batch_size in batch_list:
        for use_well in use_well_list:
            for use_ref in use_ref_list:
                main(batch_size, use_well, use_ref)