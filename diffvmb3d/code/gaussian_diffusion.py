"""
Gaussian diffusion utilities for DiffVMB3D: Depth-progressive 3D velocity model
building via 2D generative diffusion models.

This module implements the complete Gaussian diffusion framework used for training
and sampling in the depth-progressive VMB pipeline. It is adapted from Ho et al.'s
original diffusion implementation with the following DiffVMB3D-specific extensions:

  1. All model calls accept the full set of conditioning inputs defined in the paper:
     shallow velocity patch (v_shallow), structural attribute (s), depth position
     scalar (d_max), well velocity (w), and well lateral position (l).

  2. The x0-prediction parameterization (ModelMeanType.START_X) is used as the
     default, corresponding to the training objective in Eq. 3-4 of the paper:
         L = E_{v_deep, epsilon, t} || v_deep - f_theta(x_t, t, v_shallow, d_max, w, s) ||^2

  3. DDIM sampling (Song et al., 2020a) with a reduced number of denoising steps
     (10 steps in the paper) is used to accelerate inference while maintaining
     quality comparable to the full 1000-step DDPM chain.

  4. Optional well-log gradient guidance can be applied during sampling to steer
     the generated velocity toward consistency with well measurements.

Original source:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined variance schedule {beta_t}_{t=1}^{T} by name.

    The beta schedule controls the rate of noise injection in the forward
    diffusion process (Eq. 1). Two schedules are supported:
      - "linear": linearly spaced betas from beta_start to beta_end, scaled
        to remain consistent across different values of T.
      - "cosine": derived from a cosine-based alpha_bar function, which
        provides a more gradual noise schedule and is generally preferred
        for image-like data.

    Args:
        schedule_name:            Name of the schedule ("linear" or "cosine").
        num_diffusion_timesteps:  Total number of diffusion timesteps T.

    Returns:
        A 1-D numpy array of beta values of length T.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al., extended to work for any number of
        # diffusion steps by rescaling the endpoints proportionally.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Construct a beta schedule from a cumulative product function alpha_bar(t).

    Given a function alpha_bar(t) that defines the cumulative product
    ᾱ_t = prod_{s=1}^{t} (1 - beta_s) over normalized time t in [0, 1],
    this function recovers the discrete beta values by inverting the
    cumulative product at each timestep.

    Args:
        num_diffusion_timesteps:  Number of betas to produce (T).
        alpha_bar:                A callable mapping t in [0, 1] to ᾱ_t.
        max_beta:                 Upper bound on each beta to prevent
                                  numerical singularities near t = T.

    Returns:
        A 1-D numpy array of beta values of length T.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Specifies what quantity the denoising network f_theta predicts.

    In DiffVMB3D, START_X (x0-prediction) is used: the network directly
    predicts the clean velocity patch v_deep (= x_0) from the noised
    input x_t, as described in Eq. 3 of the paper.
    """
    PREVIOUS_X = enum.auto()  # Predict x_{t-1} directly
    START_X = enum.auto()     # Predict x_0 (used in DiffVMB3D, Eq. 3)
    EPSILON = enum.auto()     # Predict the noise epsilon


class ModelVarType(enum.Enum):
    """
    Specifies how the model's output variance is determined.

    In DiffVMB3D, FIXED_SMALL is used (the posterior variance from the
    forward process), since the model only predicts the mean.
    """
    LEARNED = enum.auto()        # Model outputs log-variance directly
    FIXED_SMALL = enum.auto()    # Use posterior variance (default for DiffVMB3D)
    FIXED_LARGE = enum.auto()    # Use beta_t as variance
    LEARNED_RANGE = enum.auto()  # Model outputs interpolation between FIXED_SMALL and FIXED_LARGE


class LossType(enum.Enum):
    """
    Specifies the training loss function.

    In DiffVMB3D, MSE is used: a simple mean squared error between the
    model prediction and the target (x_0 under x0-prediction, or epsilon
    under epsilon-prediction), corresponding to Eq. 3-4.
    """
    MSE = enum.auto()           # Raw MSE loss (used in DiffVMB3D)
    RESCALED_MSE = enum.auto()  # MSE with rescaled VLB for learned variance
    KL = enum.auto()            # Variational lower-bound (KL divergence)
    RESCALED_KL = enum.auto()   # KL rescaled to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Core Gaussian diffusion process for DiffVMB3D training and sampling.

    This class manages the complete diffusion pipeline:
      - Forward process: progressively corrupt clean velocity patches x_0 with
        Gaussian noise according to the variance schedule {beta_t} (Eq. 1-2).
      - Training: compute the x0-prediction MSE loss (Eq. 3-4) at randomly
        sampled timesteps.
      - Reverse process (sampling): iteratively denoise from x_T ~ N(0, I) back
        to a clean velocity prediction, using either the full DDPM chain or
        the accelerated DDIM sampler (10 steps in the paper).
      - Optional well-log gradient guidance during sampling.

    All model calls pass the full conditioning set from the depth-progressive
    framework: (x_t, v_shallow, s, d_max, w, l, t).

    Args:
        betas:              1-D numpy array of noise schedule values {beta_t}_{t=1}^{T}.
        model_mean_type:    What the model predicts (START_X for x0-prediction).
        model_var_type:     How variance is determined (FIXED_SMALL for DiffVMB3D).
        loss_type:          Training loss type (MSE for DiffVMB3D).
        rescale_timesteps:  If True, scale timesteps to [0, 1000] before passing
                            to the model (for compatibility with pretrained models).
        use_wellguide:      If True, enable optional well-log gradient guidance
                            during sampling.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        use_wellguide=True,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.use_wellguide = use_wellguide

        # Store betas in float64 for numerical precision in precomputed quantities.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # Precompute quantities derived from the variance schedule.
        # alpha_t = 1 - beta_t
        alphas = 1.0 - betas
        # ᾱ_t = prod_{s=1}^{t} alpha_s  (cumulative product, used in Eq. 2)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # ᾱ_{t-1} with ᾱ_0 = 1 (used in posterior computation)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # ᾱ_{t+1} with ᾱ_{T+1} = 0 (used for next-step predictions)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # Precomputed coefficients for the forward diffusion q(x_t | x_0) (Eq. 2):
        #   x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * epsilon
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # Coefficients for recovering x_0 from x_t and epsilon:
        #   x_0 = sqrt(1/ᾱ_t) * x_t - sqrt(1/ᾱ_t - 1) * epsilon
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Precomputed coefficients for the posterior distribution
        # q(x_{t-1} | x_t, x_0), used in DDPM reverse sampling:
        #   posterior_variance = beta_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clipped log-variance: the posterior variance is 0 at t=0, so we clip
        # the log by using the t=1 value for the t=0 entry to avoid log(0).
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # Posterior mean coefficients:
        #   mu_posterior = coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # MSE criterion used for well-log guidance loss computation.
        self.criterion = th.nn.MSELoss()

    def q_mean_variance(self, x_start, t):
        """
        Compute the mean, variance, and log-variance of the forward distribution
        q(x_t | x_0) at timestep t.

        This corresponds to the closed-form marginal (Eq. 2):
            q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)

        Args:
            x_start:  Clean data x_0 (target deep velocity patch v_deep).
                      Shape: [B, nz, ny, nx].
            t:        Diffusion timestep indices. Shape: [B].

        Returns:
            Tuple of (mean, variance, log_variance), each of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward diffusion process q(x_t | x_0) using the
        reparameterization trick (Eq. 2):

            x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * epsilon,  epsilon ~ N(0, I)

        In the DiffVMB3D context, x_start is the clean deep velocity patch v_deep,
        and x_t is the noised version that will be passed to the U-Net along with
        the shallow velocity condition v_shallow and other conditioning inputs.

        Args:
            x_start:  Clean data x_0 (v_deep). Shape: [B, nz, ny, nx].
            t:        Diffusion timestep indices. Shape: [B].
            noise:    Optional pre-generated Gaussian noise. If None, sampled
                      from N(0, I).

        Returns:
            Noised sample x_t of the same shape as x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the true posterior distribution
        q(x_{t-1} | x_t, x_0), which is tractable and Gaussian:

            q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_posterior, sigma_posterior^2 I)

        where:
            mu_posterior = coef1 * x_0 + coef2 * x_t
            sigma_posterior^2 = beta_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)

        This posterior is used in DDPM sampling as the target for the learned
        reverse distribution p_theta(x_{t-1} | x_t).

        Args:
            x_start:  Predicted or true x_0. Shape: [B, nz, ny, nx].
            x_t:      Noised sample at timestep t. Same shape.
            t:        Timestep indices. Shape: [B].

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, cond_top, struc, db, well, well_loc, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Compute the mean, variance, and predicted x_0 from the learned reverse
        distribution p_theta(x_{t-1} | x_t) by applying the denoising U-Net.

        The model f_theta receives the full set of conditioning inputs from the
        depth-progressive framework (Section III):
            f_theta(x_t, v_shallow, s, d_max, w, l, t)

        Under the x0-prediction parameterization (ModelMeanType.START_X), the
        model directly outputs the predicted clean velocity patch v_deep (= x_0).
        The reverse distribution mean is then computed analytically from the
        predicted x_0 and the current x_t using the posterior formula.

        Args:
            model:          The 2D U-Net denoising network f_theta.
            x:              Noised sample x_t. Shape: [B, nz, ny, nx].
            cond_top:       Shallow velocity patch v_shallow. Shape: [B, nz, ny, nx].
            struc:          Structural attribute s, or None if dropped.
                            Shape: [B, nz, ny, nx].
            db:             Depth position scalar d_max. Shape: [B].
            well:           Well velocity w (expanded to patch dims), or None.
                            Shape: [B, nz, ny, nx].
            well_loc:       Well lateral position l = (x, y), or None. Shape: [B, 2].
            t:              Timestep indices. Shape: [B].
            clip_denoised:  If True, clamp predicted x_0 to [-1, 1].
            denoised_fn:    Optional function applied to x_0 prediction before use.
            model_kwargs:   Additional keyword arguments passed to the model.

        Returns:
            Dict with keys:
                'mean':         Model mean for p(x_{t-1} | x_t).
                'variance':     Model variance.
                'log_variance': Log of model variance.
                'pred_xstart':  Predicted x_0 (clean velocity patch).
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        # Forward pass through the conditional U-Net with all conditioning inputs.
        # The model receives: (x_t, v_shallow, s, d_max, w, l, t).
        model_output = model(x, cond_top, struc, db, well, well_loc, self._scale_timesteps(t), **model_kwargs)

        # Handle learned variance models (not used in default DiffVMB3D config).
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # Model outputs both mean and variance in a doubled-channel tensor.
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                # Interpolate between posterior variance (min) and beta (max).
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # model_var_values in [-1, 1] maps to [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)

        else:
            # Fixed variance (FIXED_SMALL or FIXED_LARGE).
            # DiffVMB3D uses FIXED_SMALL (posterior variance).
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            """Apply optional denoised_fn and clip to [-1, 1] if requested."""
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # Model directly predicts x_{t-1}; recover x_0 from it.
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                # x0-prediction (used in DiffVMB3D): model directly outputs x_0.
                pred_xstart = process_xstart(model_output)
            else:
                # Epsilon-prediction: recover x_0 from predicted noise.
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # Compute the analytic posterior mean from predicted x_0 and x_t.
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Recover x_0 from x_t and predicted noise epsilon using the inverse
        of Eq. 2:
            x_0 = (1 / sqrt(ᾱ_t)) * x_t - sqrt(1/ᾱ_t - 1) * epsilon
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        Recover x_0 from x_t and predicted x_{t-1} by inverting the posterior
        mean formula:
            x_0 = (x_{t-1} - coef2 * x_t) / coef1
        """
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Recover the noise epsilon from x_t and predicted x_0 using Eq. 2:
            epsilon = (sqrt(1/ᾱ_t) * x_t - x_0) / sqrt(1/ᾱ_t - 1)

        This is used in DDIM sampling to convert from x0-prediction to the
        epsilon needed for the DDIM update rule.
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        Optionally rescale integer timesteps to the [0, 1000] range expected
        by models trained with a fixed 1000-step schedule.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def well_loss(self, x, well, well_mask):
        """
        Compute the well-log velocity mismatch loss at the well location.

        Measures the MSE between the generated velocity and the well-log
        velocity at the masked well positions:
            L_well = || x * mask_well - w ||^2

        Args:
            x:          Current sample (predicted velocity). Shape: [B, nz, ny, nx].
            well:       Well velocity values. Shape: [B, nz, ny, nx].
            well_mask:  Binary mask indicating the well location. Same shape.

        Returns:
            Scalar loss value (detached from computational graph).
        """
        loss = self.criterion(x * well_mask, well)
        return loss.item()

    def well_guidance(self, x, well, well_mask):
        """
        Compute the gradient of the well-log mismatch loss with respect to
        the current sample x for well-log gradient guidance during sampling.

        This enables optional posterior guidance: at each reverse diffusion step,
        the sample is nudged toward better consistency with the well-log velocity
        by taking a gradient step on the well mismatch loss:
            x <- x - (scale_factor / 2) * sqrt(ᾱ_t) * grad_x L_well

        Args:
            x:          Current sample (requires_grad will be enabled). Shape: [B, nz, ny, nx].
            well:       Well velocity values. Same shape.
            well_mask:  Binary mask for the well location. Same shape.

        Returns:
            Tuple of (gradient tensor, scalar loss value).
        """
        with th.enable_grad():
            x.requires_grad_(True)
            loss = self.criterion(x * well_mask, well)
            grad = th.autograd.grad(loss, x, retain_graph=True)[0]
        return grad, loss.item()

    def p_sample(
        self, model, x, cond_top, struc, db, well, well_loc, well_mask, t,
        scale_factor=None, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Perform a single DDPM reverse step: sample x_{t-1} from x_t.

        Implements the standard DDPM sampling equation:
            x_{t-1} = mu_theta(x_t, t) + sigma_t * z,  z ~ N(0, I)
        where mu_theta and sigma_t are derived from the model's prediction
        of x_0 (Section II). No noise is added at the final step (t = 0).

        Optionally applies well-log gradient guidance to steer the sample
        toward consistency with well measurements.

        Args:
            model:          The conditional denoising U-Net f_theta.
            x:              Current noised sample x_t. Shape: [B, nz, ny, nx].
            cond_top:       Shallow velocity patch v_shallow.
            struc:          Structural attribute s, or None.
            db:             Depth position scalar d_max.
            well:           Well velocity w, or None.
            well_loc:       Well lateral position l, or None.
            well_mask:      Binary mask for the well location.
            t:              Current timestep indices. Shape: [B].
            scale_factor:   If not None, strength of well-log gradient guidance.
            clip_denoised:  If True, clip predicted x_0 to [-1, 1].
            denoised_fn:    Optional function applied to x_0 before sampling.
            model_kwargs:   Additional keyword arguments for the model.

        Returns:
            Dict with keys: 'sample' (x_{t-1}), 'pred_xstart' (predicted x_0),
            'loss_before' and 'loss_after' (well mismatch before/after guidance).
        """
        out = self.p_mean_variance(
            model,
            x,
            cond_top, struc, db, well, well_loc,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # Mask to suppress noise injection at the final step (t = 0).
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        # DDPM sampling: x_{t-1} = mean + sigma * noise
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # Compute well mismatch loss before guidance (for monitoring).
        if well is not None:
            loss_before = self.well_loss(sample, well, well_mask)
        else:
            loss_before = 0.0
        loss_after = 0.0

        # Optional well-log gradient guidance: nudge the sample toward better
        # well consistency by following the negative gradient of the well loss.
        if scale_factor is not None:
            cond_grad, _ = self.well_guidance(sample, well, well_mask)
            sample = sample - scale_factor * th.exp(0.5 * out["log_variance"]) * cond_grad
            loss_after = self.well_loss(sample, well, well_mask)

        # Print well guidance monitoring at every 5th timestep and the final step.
        if (t[0].item() + 1) % 5 == 0 or t[0].item() == 0:
            print(f'Time step {t[0].item()} --> Loss before {loss_before} and Loss after {loss_after}')

        return {"sample": sample, "pred_xstart": out["pred_xstart"], 'loss_before': loss_before, 'loss_after': loss_after}

    def p_sample_loop(
        self,
        model,
        cond_top, struc, db, well, well_loc, well_mask,
        shape,
        noise=None,
        scale_factor=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate velocity samples using the full DDPM reverse chain.

        Iterates from t = T-1 down to t = 0, calling p_sample at each step
        to progressively denoise from pure Gaussian noise x_T ~ N(0, I) to
        a clean velocity prediction. This is the standard DDPM sampling
        procedure (not used in the paper's experiments, which use DDIM instead).

        Args:
            model:       The conditional denoising U-Net f_theta.
            cond_top:    Shallow velocity patch v_shallow.
            struc:       Structural attribute s, or None.
            db:          Depth position scalar d_max.
            well:        Well velocity w, or None.
            well_loc:    Well lateral position l, or None.
            well_mask:   Binary mask for the well location.
            shape:       Output sample shape (B, nz, ny, nx).
            noise:       Optional initial noise; if None, sampled from N(0, I).
            scale_factor: Strength of optional well gradient guidance.
            clip_denoised: If True, clip predicted x_0 to [-1, 1].
            denoised_fn: Optional function applied to x_0 predictions.
            model_kwargs: Additional keyword arguments for the model.
            device:      Device for tensor creation.
            progress:    If True, display a tqdm progress bar.

        Returns:
            Tuple of (final_sample, intermediate_samples, predicted_x0s,
                      loss_before_list, loss_after_list).
        """
        final = None
        for sample, image_all, pred_xstart, loss_before, loss_after, in self.p_sample_loop_progressive(
            model,
            cond_top, struc, db, well, well_loc, well_mask,
            shape,
            noise=noise,
            scale_factor=scale_factor,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final, image_all, pred_xstart, loss_before, loss_after = sample, image_all, pred_xstart, loss_before, loss_after
        return final["sample"], image_all, pred_xstart, loss_before, loss_after

    def p_sample_loop_progressive(
        self,
        model,
        cond_top, struc, db, well, well_loc, well_mask,
        shape,
        noise=None,
        scale_factor=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generator that yields intermediate results at each DDPM reverse step.

        Iterates from t = T-1 to t = 0. At each step, yields the current
        sample, accumulated intermediate snapshots (every 50 steps), predicted
        x_0, and well guidance losses.

        Args:
            Same as p_sample_loop().

        Yields:
            Tuple of (out_dict, intermediate_samples, pred_xstart_list,
                      loss_before_list, loss_after_list) at each timestep.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            # Start from pure Gaussian noise: x_T ~ N(0, I).
            image = th.randn(*shape, device=device)

        # Iterate from t = T-1 down to t = 0 (reverse diffusion).
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []      # Intermediate sample snapshots (every 50 steps)
        pred_xstart = []    # Corresponding x_0 predictions
        loss_before = []    # Well loss before guidance at each step
        loss_after = []     # Well loss after guidance at each step
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    image,
                    cond_top, struc, db, well, well_loc, well_mask,
                    t,
                    scale_factor=scale_factor,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                # Save intermediate snapshots every 50 timesteps for visualization.
                if (i + 1) % 50 == 0:
                    image_all.append(out["sample"])
                    pred_xstart.append(out["pred_xstart"])
                loss_before.append(out["loss_before"])
                loss_after.append(out["loss_after"])
                yield out, image_all, pred_xstart, loss_before, loss_after
                image = out["sample"]

    # =====================================================================
    #  DDIM Sampling (Denoising Diffusion Implicit Models)
    #
    #  DDIM (Song et al., 2020a) enables accelerated sampling by skipping
    #  timesteps in the reverse chain. In the paper, DDIM with 10 denoising
    #  steps replaces the full 1000-step DDPM chain, reducing inference time
    #  substantially while producing results of comparable quality.
    #
    #  When eta = 0, DDIM sampling is fully deterministic (no stochasticity
    #  in the reverse process), which makes inference reproducible.
    # =====================================================================

    def ddim_sample(
        self,
        model,
        x,
        cond_top, struc, db, well, well_loc, well_mask,
        t,
        scale_factor=None, 
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Perform a single DDIM reverse step: sample x_{t-1} from x_t.

        Implements the DDIM update rule (Song et al., 2020a, Equation 12):
            x_{t-1} = sqrt(ᾱ_{t-1}) * pred_x0
                    + sqrt(1 - ᾱ_{t-1} - sigma^2) * epsilon_theta
                    + sigma * noise

        where sigma controls the stochasticity:
            sigma = eta * sqrt((1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * sqrt(1 - ᾱ_t / ᾱ_{t-1})

        When eta = 0 (default in DiffVMB3D), the process is fully deterministic.

        Args:
            model:          The conditional denoising U-Net f_theta.
            x:              Current noised sample x_t. Shape: [B, nz, ny, nx].
            cond_top:       Shallow velocity patch v_shallow.
            struc:          Structural attribute s, or None.
            db:             Depth position scalar d_max.
            well:           Well velocity w, or None.
            well_loc:       Well lateral position l, or None.
            well_mask:      Binary mask for the well location.
            t:              Current timestep indices. Shape: [B].
            scale_factor:   If not None, strength of well gradient guidance.
            clip_denoised:  If True, clip predicted x_0 to [-1, 1].
            denoised_fn:    Optional function applied to x_0 predictions.
            model_kwargs:   Additional keyword arguments for the model.
            eta:            DDIM stochasticity parameter (0 = deterministic,
                            1 = equivalent to DDPM).

        Returns:
            Dict with keys: 'sample', 'pred_xstart', 'loss_before', 'loss_after'.
        """
        out = self.p_mean_variance(
            model,
            x,
            cond_top, struc, db, well, well_loc,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Derive epsilon from the predicted x_0, regardless of the model's
        # native parameterization (x0-prediction or epsilon-prediction).
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        # DDIM noise level: sigma = 0 when eta = 0 (deterministic sampling).
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # DDIM update (Equation 12 in Song et al., 2020a):
        #   mean_pred = sqrt(ᾱ_{t-1}) * x0_pred + sqrt(1 - ᾱ_{t-1} - sigma^2) * eps
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        # No stochastic noise at the final step (t = 0).
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise

        # Well-log mismatch monitoring before optional guidance.
        if well is not None:
            loss_before = self.well_loss(sample, well, well_mask)
        else:
            loss_before = 0.0
        loss_after = 0.0

        # Optional well-log gradient guidance (same as in p_sample).
        if scale_factor is not None:
            cond_grad, _ = self.well_guidance(sample, well, well_mask)
            sample = sample - scale_factor * th.exp(0.5 * out["log_variance"]) * cond_grad
            loss_after = self.well_loss(sample, well, well_mask)

        if (t[0].item() + 1) % 5 == 0 or t[0].item() == 0:
            print(f'Time step {t[0].item()} --> Loss before {loss_before} and Loss after {loss_after}')

        return {"sample": sample, "pred_xstart": out["pred_xstart"], 'loss_before': loss_before, 'loss_after': loss_after}

    def ddim_sample_loop(
        self,
        model,
        cond_top, struc, db, well, well_loc, well_mask,
        shape,
        noise=None,
        scale_factor=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate velocity samples using the DDIM reverse chain.

        This is the primary sampling method used in DiffVMB3D inference (Section
        III.II, Eq. 7). With 10 denoising steps (via a timestep-respaced diffusion
        schedule), DDIM produces results comparable to the full 1000-step DDPM chain
        at a fraction of the computational cost.

        Fifty independent samples are generated per conditioning scenario, and the
        ensemble mean is used as the final prediction (Section IV).

        Args:
            Same as p_sample_loop(), plus:
            eta:  DDIM stochasticity parameter (0 = deterministic).

        Returns:
            Tuple of (final_sample, intermediate_samples, predicted_x0s,
                      loss_before_list, loss_after_list).
        """
        final = None
        for sample, image_all, pred_xstart, loss_before, loss_after in self.ddim_sample_loop_progressive(
            model,
            cond_top, struc, db, well, well_loc, well_mask,
            shape,
            noise=noise,
            scale_factor=scale_factor,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final, image_all, pred_xstart, loss_before, loss_after = sample, image_all, pred_xstart, loss_before, loss_after
        return final["sample"], image_all, pred_xstart, loss_before, loss_after

    def ddim_sample_loop_progressive(
        self,
        model,
        cond_top, struc, db, well, well_loc, well_mask,
        shape,
        noise=None,
        scale_factor=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generator that yields intermediate results at each DDIM reverse step.

        Iterates over the (potentially subsampled) timestep schedule from
        t = T-1 to t = 0. When using timestep respacing (e.g., 'ddim10'),
        only 10 steps are executed instead of the full 1000.

        Args:
            Same as ddim_sample_loop().

        Yields:
            Tuple of (out_dict, intermediate_samples, pred_xstart_list,
                      loss_before_list, loss_after_list) at each timestep.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            # Start from pure Gaussian noise: x_T ~ N(0, I).
            image = th.randn(*shape, device=device)

        # Reverse timestep indices (T-1, T-2, ..., 0).
        # With timestep respacing (e.g., ddim10), num_timesteps = 10.
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []      # Intermediate sample snapshots
        pred_xstart = []    # Corresponding x_0 predictions
        loss_before = []    # Well loss before guidance at each step
        loss_after = []     # Well loss after guidance at each step
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                        model,
                        image,
                        cond_top, struc, db, well, well_loc, well_mask,
                        t,
                        scale_factor=scale_factor,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                )
            # Save intermediate snapshots every 50 original timesteps.
            if i % 50 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])
            loss_before.append(out["loss_before"])
            loss_after.append(out["loss_after"])
            yield out, image_all, pred_xstart, loss_before, loss_after
            image = out["sample"]

    # =====================================================================
    #  Variational lower-bound (VLB) utilities
    #
    #  These methods compute the variational bound and related quantities,
    #  primarily used for evaluation and for training with learned variance.
    #  They are not used in the default DiffVMB3D configuration (which uses
    #  MSE loss with fixed variance), but are retained for completeness.
    # =====================================================================

    def _vb_terms_bpd(
        self, model, x_start, x_t, cond_top, struc, db, well, well_loc, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Compute a single term of the variational lower-bound in bits-per-dim.

        At t = 0, returns the decoder negative log-likelihood.
        At t > 0, returns KL(q(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t)).

        Args:
            model:       The denoising model (or a frozen output for VLB-only).
            x_start:     Clean data x_0. Shape: [B, nz, ny, nx].
            x_t:         Noised data at timestep t. Same shape.
            cond_top:    Shallow velocity patch v_shallow.
            struc:       Structural attribute s, or None.
            db:          Depth position scalar d_max.
            well:        Well velocity w, or None.
            well_loc:    Well lateral position l, or None.
            t:           Timestep indices. Shape: [B].
            clip_denoised: If True, clip predicted x_0 to [-1, 1].
            model_kwargs:  Additional keyword arguments.

        Returns:
            Dict with 'output' (per-sample NLL/KL in bits) and 'pred_xstart'.
        """

        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )

        out = self.p_mean_variance(
            model, x_t, cond_top, struc, db, well, well_loc, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At t = 0, return decoder NLL; otherwise return KL divergence.
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, cond_top, struc, db, well, well_loc, t, model_kwargs=None, noise=None):
        """
        Compute the training loss for a single diffusion timestep.

        This is the core training objective of DiffVMB3D (Eq. 3-4). For the
        default MSE loss with x0-prediction:
          1. Sample noise: epsilon ~ N(0, I)
          2. Construct noised input: x_t = sqrt(ᾱ_t) * v_deep + sqrt(1-ᾱ_t) * epsilon
          3. Forward pass: pred = f_theta(x_t, v_shallow, s, d_max, w, l, t)
          4. Compute loss: L = || v_deep - pred ||^2

        The conditioning inputs (v_shallow, s, d_max, w, l) are passed directly
        to the model. Classifier-free guidance dropout (setting s or w to None)
        is handled upstream in train_util.forward_backward().

        Args:
            model:       The 2D U-Net denoising network f_theta.
            x_start:     Clean target v_deep (= x_0). Shape: [B, nz, ny, nx].
            cond_top:    Shallow velocity patch v_shallow. Shape: [B, nz, ny, nx].
            struc:       Structural attribute s, or None (dropped by CFG).
            db:          Depth position scalar d_max. Shape: [B].
            well:        Well velocity w, or None (dropped by CFG).
            well_loc:    Well lateral position l, or None (dropped by CFG).
            t:           Sampled diffusion timesteps. Shape: [B].
            model_kwargs: Additional keyword arguments for the model.
            noise:       Optional pre-generated noise; if None, sampled from N(0, I).

        Returns:
            Dict with key "loss" containing a per-sample loss tensor of shape [B].
            May also contain "mse" and optionally "vb" depending on the loss type.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Forward diffusion: corrupt x_0 to x_t using Eq. 2.
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            # Variational lower-bound loss (not used in default DiffVMB3D).
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                cond_top=cond_top,
                struc=struc,
                db=db,
                well=well,
                well_loc=well_loc,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # === MSE loss (default for DiffVMB3D) ===
            # Forward pass through the conditional U-Net:
            #   model(x_t, v_shallow, s, d_max, w, l, t)
            model_output = model(x_t, cond_top, struc, db, well, well_loc, self._scale_timesteps(t), **model_kwargs)

            # Handle learned variance (not used in default DiffVMB3D).
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])

                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance via VLB, but detach the mean so that the
                # variance loss does not affect the mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    cond_top=cond_top,
                    struc=struc,
                    db=db,
                    well=well,
                    well_loc=well_loc,
                    t=t,
                    clip_denoised=False,
                )["output"]

                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            # Determine the prediction target based on model parameterization:
            #   - START_X (x0-prediction, used in DiffVMB3D): target = x_0 = v_deep
            #   - EPSILON: target = noise epsilon
            #   - PREVIOUS_X: target = posterior mean of q(x_{t-1} | x_t, x_0)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            # Compute per-sample MSE: || target - f_theta(...) ||^2
            # For x0-prediction: || v_deep - f_theta(x_t, t, v_shallow, d_max, w, s) ||^2
            terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Compute the prior KL term of the variational lower-bound in bits-per-dim.

        This measures KL(q(x_T | x_0) || N(0, I)), which depends only on the
        forward process and is not optimized. Used for VLB evaluation.

        Args:
            x_start:  Clean data x_0. Shape: [B, nz, ny, nx].

        Returns:
            Per-sample KL values in bits. Shape: [B].
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, cond_top, struc, db, well, well_loc, clip_denoised=True, model_kwargs=None):
        """
        Compute the full variational lower-bound (VLB) across all timesteps,
        measured in bits-per-dim.

        Evaluates each term of the VLB individually from t = T-1 to t = 0,
        summing them to obtain the total bound. Also computes per-timestep
        x_0 MSE and epsilon MSE for diagnostic purposes.

        This method is primarily used for evaluation, not during standard
        DiffVMB3D training.

        Args:
            model:       The denoising model.
            x_start:     Clean data x_0.
            cond_top:    Shallow velocity patch v_shallow.
            struc:       Structural attribute s.
            db:          Depth position scalar d_max.
            well:        Well velocity w.
            well_loc:    Well lateral position l.
            clip_denoised: If True, clip predicted x_0 to [-1, 1].
            model_kwargs:  Additional keyword arguments.

        Returns:
            Dict with keys:
                'total_bpd':   Total VLB per sample. Shape: [B].
                'prior_bpd':   Prior KL term per sample. Shape: [B].
                'vb':          Per-timestep VLB terms. Shape: [B, T].
                'xstart_mse':  Per-timestep x_0 MSE. Shape: [B, T].
                'mse':         Per-timestep epsilon MSE. Shape: [B, T].
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Compute the VLB term at the current timestep.
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    cond_top=cond_top,
                    struc=struc,
                    db=db,
                    well=well,
                    well_loc=well_loc,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Index into a 1-D numpy array using a batch of integer timestep indices,
    and broadcast the result to a target shape.

    This utility is used throughout the diffusion process to extract
    timestep-specific coefficients (e.g., sqrt(ᾱ_t), beta_t) and
    expand them to match the spatial dimensions of the data tensors.

    Args:
        arr:             1-D numpy array of precomputed coefficients (length T).
        timesteps:       Integer tensor of timestep indices. Shape: [B].
        broadcast_shape: Target shape (B, C, H, W) for broadcasting.

    Returns:
        Tensor of shape broadcast_shape with the indexed values.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)