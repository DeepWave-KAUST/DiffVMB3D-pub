"""
Timestep respacing for accelerated diffusion sampling in DiffVMB3D.

This module enables running the reverse diffusion process with fewer timesteps
than the original training schedule (T = 1000), which is essential for efficient
inference. In the paper (Section IV), DDIM sampling with only 10 denoising steps
is used instead of the full 1000-step DDPM chain, reducing inference time by ~100x
while maintaining comparable generation quality.

The key idea is to select a subset of S timesteps {t_1, t_2, ..., t_S} from the
original T-step schedule and recompute effective betas for this reduced schedule
such that the marginal distributions q(x_{t_i} | x_0) are preserved exactly.
This means the model trained on the full schedule can be used directly for
sampling with the reduced schedule without any retraining.

Usage example for DiffVMB3D inference:
    # "ddim10" selects 10 evenly-spaced timesteps from the original 1000
    timestep_respacing = "ddim10"
    use_timesteps = space_timesteps(1000, timestep_respacing)
    diffusion = SpacedDiffusion(use_timesteps=use_timesteps, betas=betas, ...)
    # Now diffusion.ddim_sample_loop() runs only 10 reverse steps
"""

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Select a subset of timesteps from an original T-step diffusion schedule.

    Supports two modes:
      1. Section-based striding: Divide the T timesteps into equal sections and
         select a specified number of steps from each section. For example, with
         T=300 and section_counts=[10, 15, 20], the first 100 steps yield 10
         evenly-spaced indices, the next 100 yield 15, and the final 100 yield 20.

      2. DDIM striding (prefix "ddim"): Select N evenly-spaced timesteps using a
         fixed integer stride, matching the convention from Song et al. (2020a).
         For DiffVMB3D, "ddim10" selects 10 timesteps from the original 1000-step
         schedule (stride = 100), enabling fast inference as described in Section IV.

    Args:
        num_timesteps:   Total number of diffusion steps T in the original schedule.
        section_counts:  Either:
                         - A string "ddimN" for DDIM-style uniform striding with N steps.
                         - A string of comma-separated integers "N1,N2,..." for
                           per-section step counts.
                         - A list of integers for per-section step counts.

    Returns:
        A set of integer timestep indices from the original schedule to retain.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # DDIM striding: find an integer stride that yields exactly N steps.
            # For T=1000 and N=10, stride=100 -> {0, 100, 200, ..., 900}.
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    # Divide the T timesteps into len(section_counts) equal-sized sections,
    # distributing any remainder to the first sections.
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        # Compute fractional stride to evenly space section_count indices
        # within the current section of `size` timesteps.
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that operates on a subset of the original T timesteps,
    enabling accelerated sampling (e.g., 10-step DDIM for DiffVMB3D inference).

    This class recomputes the beta schedule for the reduced timestep set so that
    the marginal noise distributions q(x_{t_i} | x_0) match exactly:

        beta'_i = 1 - ᾱ_{t_i} / ᾱ_{t_{i-1}}

    where ᾱ_{t_i} is the cumulative product of alphas at the original timestep t_i,
    and ᾱ_{t_0} = 1. This ensures that a model trained on the full schedule can be
    used without modification for sampling on the reduced schedule.

    The model is wrapped by _WrappedModel to map the reduced timestep indices
    (0, 1, ..., S-1) back to the original schedule indices before passing them
    to the U-Net's sinusoidal timestep embedding.

    Args:
        use_timesteps:  A set of timestep indices from the original schedule to
                        retain (output of space_timesteps()).
        **kwargs:       Arguments for the base GaussianDiffusion constructor,
                        including the original full beta schedule.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        # Mapping from reduced indices {0, ..., S-1} to original indices {t_1, ..., t_S}.
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # Create a temporary base diffusion to access the full ᾱ_t schedule.
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0

        # Recompute betas for the selected timesteps.
        # For each selected timestep t_i, the effective beta is:
        #   beta'_i = 1 - ᾱ_{t_i} / ᾱ_{t_{i-1}}
        # This preserves the marginal: q(x_{t_i} | x_0) remains N(sqrt(ᾱ_{t_i}) x_0, (1-ᾱ_{t_i}) I).
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # Replace the original betas with the reduced schedule and initialize
        # the parent GaussianDiffusion with S steps instead of T.
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        """Override to wrap the model with timestep remapping before inference."""
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        """Override to wrap the model with timestep remapping during training."""
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        """
        Wrap the U-Net model to automatically remap reduced timestep indices
        to the original schedule before computing sinusoidal embeddings.
        Avoids double-wrapping if already wrapped.
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        """
        Identity scaling: timestep rescaling is handled by the wrapped model
        instead, since it needs access to the original timestep indices.
        """
        return t


class _WrappedModel:
    """
    Wrapper around the U-Net model that remaps reduced timestep indices to
    the original diffusion schedule before passing them to the network.

    When using SpacedDiffusion with S < T timesteps, the diffusion process
    internally uses indices {0, 1, ..., S-1}. However, the U-Net's sinusoidal
    timestep embedding was trained on the original indices {0, 1, ..., T-1}.
    This wrapper maps each reduced index back to its corresponding original
    index via timestep_map[i], so the model receives the correct timestep
    encoding.

    For DiffVMB3D with T=1000 and S=10 (ddim10):
        reduced index 0 -> original index 0
        reduced index 1 -> original index 100
        ...
        reduced index 9 -> original index 900

    Args:
        model:               The original U-Net denoising network f_theta.
        timestep_map:        List mapping reduced indices to original indices.
        rescale_timesteps:   If True, rescale original indices to [0, 1000].
        original_num_steps:  The original number of diffusion steps T.
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, cond_top, struc, db, well, well_loc, ts, **kwargs):
        """
        Forward pass with timestep remapping.

        Maps the reduced timestep indices to original indices, optionally
        rescales them, then calls the U-Net with the full conditioning set:
            f_theta(x_t, v_shallow, s, d_max, w, l, t_original)

        Args:
            x:         Noised sample x_t. Shape: [B, nz, ny, nx].
            cond_top:  Shallow velocity embedding c_emb.
            struc:     Structural attribute s_emb, or None.
            db:        Depth position d_max.
            well:      Well velocity w_emb, or None.
            well_loc:  Well position l_emb, or None.
            ts:        Reduced timestep indices {0, ..., S-1}. Shape: [B].
            **kwargs:  Additional model keyword arguments.

        Returns:
            Model prediction (predicted x_0 under x0-prediction).
        """
        # Look up the original timestep index for each sample in the batch.
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        # Optionally rescale to [0, 1000] for compatibility with models trained
        # on a fixed 1000-step schedule.
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        return self.model(x, cond_top, struc, db, well, well_loc, new_ts, **kwargs)