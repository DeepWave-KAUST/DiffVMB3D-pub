"""
Training utilities for DiffVMB3D: Depth-progressive 3D velocity model building
via 2D generative diffusion models.

This module implements the main training loop for the conditional diffusion model
that learns to predict deep velocity patches from shallow velocity patches under
the depth-as-channel formulation. The training follows the x0-prediction objective
(Eq. 3-4 in the paper) with classifier-free guidance dropout for optional well-log
and structural-attribute constraints (Section II).

Key variable naming convention throughout this module:
    batch_vp       : target deep velocity patch v_deep (x0 in diffusion notation)
    batch_cond_top : shallow velocity conditioning patch v_shallow
    batch_struc    : structural attribute s (e.g., convolution image)
    batch_db       : depth position scalar d_max (deepest grid point of v_deep)
    batch_well     : well-log velocity profile w (expanded to patch dimensions)
    batch_well_loc : lateral position of the well l = (x, y)
"""

import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import time
import random
from scipy.io import loadmat

# Initial value for the log2 loss scale used in mixed-precision (fp16) training.
# The scale is dynamically adjusted during training: increased gradually to
# improve precision, and decreased when NaN gradients are detected.
INITIAL_LOG_LOSS_SCALE = 20.0

# Directory for saving model checkpoints (including EMA weights).
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)


class TrainLoop:
    """
    Main training loop for the depth-progressive conditional diffusion model.

    This class manages the complete training procedure, including:
      - Loading/resuming from checkpoints
      - Running the forward diffusion (noising) and computing the x0-prediction
        loss (Eq. 4) at each iteration
      - Applying classifier-free guidance dropout on well-log (w) and structural
        attribute (s) constraints independently with configurable probabilities
        (wellcond_drop, refcond_drop), as described in Section II
      - Maintaining exponential moving average (EMA) of model weights for
        stable sampling at inference time
      - Optional mixed-precision (fp16) training support
      - Periodic logging and checkpoint saving

    Args:
        model:              The 2D U-Net denoising network f_theta under the
                            depth-as-channel formulation (Section III.I & III.III).
        diffusion:          The Gaussian diffusion process handler, which manages
                            the forward noising schedule {beta_t} and computes the
                            training loss (Eq. 3-4).
        data:               An iterator yielding training tuples of
                            (v_deep, v_shallow, s, d_max, w, l, cond).
        batch_size:         Number of training samples per iteration.
        microbatch:         Sub-batch size for gradient accumulation; set to -1
                            to use the full batch_size.
        lr:                 Learning rate for the AdamW optimizer (fixed at 1e-4
                            in the paper).
        ema_rate:           Decay rate(s) for exponential moving average of model
                            weights (0.999 in the paper).
        log_interval:       Number of steps between logging loss statistics.
        save_interval:      Number of steps between saving checkpoints.
        resume_checkpoint:  Path to a checkpoint file for resuming training.
        use_fp16:           Whether to use mixed-precision (float16) training.
        fp16_scale_growth:  Increment for log2 loss scale per step in fp16 mode.
        schedule_sampler:   Diffusion timestep sampler; defaults to UniformSampler
                            which draws t uniformly from {1, ..., T}.
        weight_decay:       L2 weight decay coefficient for AdamW.
        lr_anneal_steps:    If > 0, linearly anneal the learning rate to zero
                            over this many steps.
        wellcond_drop:      Dropout probability for well-log constraint w during
                            classifier-free guidance training (5% in the paper).
        refcond_drop:       Dropout probability for structural attribute s during
                            classifier-free guidance training (5% in the paper).
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        wellcond_drop=0,
        refcond_drop=0,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        # If microbatch <= 0, use the full batch without gradient accumulation.
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr

        # Support single or comma-separated EMA rates (e.g., "0.999" or "0.999,0.9999").
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # Default to uniform timestep sampling if no custom sampler is provided.
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # Classifier-free guidance dropout probabilities for well and structural
        # conditions, applied independently at each training iteration (Section II).
        self.wellcond_drop = wellcond_drop
        self.refcond_drop = refcond_drop

        self.step = 0
        self.resume_step = 0

        # Total batch size across all processes (here single-GPU, so equals batch_size).
        self.global_batch = self.batch_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Load model weights from a checkpoint if resuming training.
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        # Initialize AdamW optimizer with betas=(0.9, 0.999) as used in the paper.
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        if self.resume_step:
            # Restore optimizer state and EMA parameters from checkpoint.
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Initialize EMA parameters as copies of the current model weights.
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        """Load model parameters from a checkpoint file if one exists."""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location=self.device))

    def _load_ema_parameters(self, rate):
        """
        Load EMA parameters for a given decay rate from the corresponding
        checkpoint file (named ema_{rate}_{step}.pt).
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load_state_dict(
                ema_checkpoint, map_location=self.device
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """Load the AdamW optimizer state dict from the corresponding checkpoint."""
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Set up mixed-precision training by creating float32 master copies of
        model parameters and converting the model to float16.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        Main training loop. Iterates over the dataset, performing one training
        step per iteration until lr_anneal_steps is reached (or indefinitely if
        lr_anneal_steps == 0). Periodically logs training statistics and saves
        model checkpoints including EMA weights.
        """
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # Benchmark wall-clock time for the first 100 and 200 iterations.
            if self.step == 1:
                start = time.time()
            if self.step == 101:
                end = time.time()
                print(f'time cost {end - start} s')
            if self.step == 201:
                end = time.time()
                print(f'time cost {end - start} s')

            # Fetch the next training batch from the data iterator.
            # Each batch contains:
            #   batch_vp       : [B, nz, ny, nx]  target deep velocity patch (v_deep)
            #   batch_cond_top : [B, nz, ny, nx]  shallow conditioning patch (v_shallow)
            #   batch_struc    : [B, nz, ny, nx]  structural attribute (s)
            #   batch_db       : [B]              depth position scalar (d_max)
            #   batch_well     : [B, nz, ny, nx]  well velocity (w), expanded laterally
            #   batch_well_loc : [B, 2]           well lateral position (l = (x, y))
            #   cond           : dict             additional model keyword arguments
            batch_vp, batch_cond_top, batch_struc, batch_db, batch_well, batch_well_loc, cond = next(self.data)

            self.run_step(batch_vp, batch_cond_top, batch_struc, batch_db, batch_well, batch_well_loc, cond)

            # Periodically dump accumulated log statistics (loss, grad norm, etc.).
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            # Periodically save model and EMA checkpoints.
            if self.step % self.save_interval == 0:
                self.save()
                # Early exit for integration testing.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save a final checkpoint if the last step was not on a save boundary.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_vp, batch_cond_top, batch_struc, batch_db, batch_well, batch_well_loc, cond):
        """
        Execute a single training step: forward/backward pass, parameter update,
        and logging. Dispatches to fp16 or normal optimization path accordingly.
        """
        self.forward_backward(batch_vp, batch_cond_top, batch_struc, batch_db, batch_well, batch_well_loc, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch_vp, batch_cond_top, batch_struc, batch_db, batch_well, batch_well_loc, cond):
        """
        Compute the forward diffusion and the denoising training loss (Eq. 4),
        then backpropagate gradients.

        This method implements the core training objective of DiffVMB3D:
            L = E_{v_deep, epsilon, t} || v_deep - f_theta(x_t, t, v_shallow, d_max, w, s) ||^2

        Classifier-free guidance dropout (Section II):
            At each training iteration, the well constraint (w, l) and the structural
            attribute (s) are independently dropped (set to None) with probabilities
            wellcond_drop and refcond_drop, respectively. This teaches the network to
            handle missing constraints gracefully at inference time, enabling flexible
            conditioning scenarios (unconditional, well-only, image-only, well+image).
        """
        zero_grad(self.model_params)

        # Move all conditioning tensors to the compute device (GPU).
        batch_vp = batch_vp.to(self.device)
        batch_cond_top = batch_cond_top.to(self.device)
        batch_struc = batch_struc.to(self.device)
        batch_db = batch_db.to(self.device)
        batch_well = batch_well.to(self.device)
        batch_well_loc = batch_well_loc.to(self.device)

        # Classifier-free guidance: randomly drop the well constraint.
        # When dropped, the network learns the distribution p(v_deep | v_shallow, d_max, s)
        # without well conditioning, enabling unconditional or image-only inference.
        if random.random() < self.wellcond_drop:
            batch_well = None
            batch_well_loc = None

        # Classifier-free guidance: randomly drop the structural attribute.
        # When dropped, the network learns p(v_deep | v_shallow, d_max, w) without
        # structural guidance, enabling unconditional or well-only inference.
        if random.random() < self.refcond_drop:
            batch_struc = None

        # Sample diffusion timesteps t uniformly from {1, ..., T} and obtain
        # importance weights (uniform weights for UniformSampler).
        t, weights = self.schedule_sampler.sample(batch_vp.shape[0], self.device)

        # Compute the x0-prediction training loss (Eq. 3-4).
        # The diffusion handler internally:
        #   1. Noises v_deep to x_t using the forward process (Eq. 2)
        #   2. Passes (x_t, t, v_shallow, d_max, s, w, l) to the U-Net f_theta
        #   3. Computes || v_deep - f_theta(...) ||^2
        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                batch_vp,         # x0: target deep velocity patch (v_deep)
                batch_cond_top,   # Shallow velocity patch (v_shallow)
                batch_struc,      # Structural attribute (s), or None if dropped
                batch_db,         # Depth position scalar (d_max)
                batch_well,       # Well velocity (w), or None if dropped
                batch_well_loc,   # Well lateral position (l), or None if dropped
                t,                # Sampled diffusion timesteps
                model_kwargs=cond,
        )

        losses = compute_losses()

        # If using a loss-aware timestep sampler, update its internal statistics
        # with the per-sample losses for adaptive timestep sampling.
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
            )

        # Compute the weighted mean loss and log per-timestep-quartile statistics.
        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        # Backpropagate gradients. In fp16 mode, scale the loss to prevent
        # underflow in half-precision gradient computation.
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize_fp16(self):
        """
        Perform a parameter update step under mixed-precision training.

        Handles NaN detection in gradients (reducing the loss scale if NaN is found),
        gradient transfer from fp16 model params to fp32 master params, learning rate
        annealing, optimizer step, and EMA update.
        """
        # Check for NaN/Inf gradients; if found, skip this step and reduce loss scale.
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        # Transfer gradients from fp16 model parameters to fp32 master parameters.
        model_grads_to_master_grads(self.model_params, self.master_params)
        # Unscale gradients by dividing by the loss scale factor.
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update EMA parameters: theta_ema <- rate * theta_ema + (1 - rate) * theta.
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        # Copy updated fp32 master parameters back to fp16 model parameters.
        master_params_to_model_params(self.model_params, self.master_params)
        # Gradually increase the loss scale for better precision.
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Perform a standard (fp32) parameter update step: log gradient norm,
        optionally anneal the learning rate, run the AdamW optimizer step,
        and update EMA parameters.
        """
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update EMA parameters (decay rate = 0.999 in the paper) for stable
        # sampling during inference.
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """Compute and log the L2 norm of all model gradients for monitoring."""
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is None:
                continue
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Apply linear learning rate annealing if lr_anneal_steps > 0.
        The learning rate decays linearly from lr to 0 over lr_anneal_steps.
        """
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """Log the current training step, total samples processed, and fp16 loss scale."""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save model checkpoints. Only EMA weights are saved by default, as the
        EMA model (with decay rate 0.999) produces more stable and higher-quality
        samples during inference compared to the raw training weights.
        """
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                 filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(dir_checkpoints, filename), "wb") as f:
                th.save(state_dict, f)

        # Save EMA checkpoints for each configured EMA decay rate.
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert the flat list of master parameters back into a model state dict.
        In fp16 mode, this involves unflattening the fp32 master params to match
        the original model parameter shapes.
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Extract model parameters from a state dict and convert to master params.
        In fp16 mode, master params are stored in fp32 for numerical stability.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse the training step number from a checkpoint filename.
    Expected format: path/to/modelNNNNNN.pt, where NNNNNN is the step count.

    Returns:
        int: The parsed step number, or 0 if parsing fails.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def parse_dataname_from_filename(filename):
    """
    Parse a data identifier from a filename containing 'gaussian5'.
    Used for distinguishing datasets in multi-dataset training setups.

    Returns:
        str: The parsed data identifier, or 0 if parsing fails.
    """
    split = filename.split("gaussian5")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return split1
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Get the directory for blob storage logging. Defaults to the logger's
    current directory if DIFFUSION_BLOB_LOGDIR is not set.
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    Automatically discover the latest checkpoint for resuming training.
    Returns None by default; override this for automatic checkpoint discovery
    on cloud/blob storage infrastructure.
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Locate the EMA checkpoint file corresponding to a given main checkpoint,
    training step, and EMA decay rate.

    Args:
        main_checkpoint: Path to the main model checkpoint.
        step:            Training step number.
        rate:            EMA decay rate (e.g., 0.999).

    Returns:
        str or None: Path to the EMA checkpoint if it exists, otherwise None.
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log training loss statistics, including the overall mean and per-quartile
    breakdowns. The quartile breakdown partitions the diffusion timestep range
    [0, T] into four equal bins and reports the mean loss within each bin,
    which helps diagnose whether the model struggles more at high-noise (large t)
    or low-noise (small t) levels.

    Args:
        diffusion: The diffusion process handler (provides num_timesteps = T).
        ts:        Tensor of sampled diffusion timesteps for the current batch.
        losses:    Dict of loss tensors keyed by loss name (e.g., "loss", "mse").
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log per-quartile loss: q0 (t near 0, low noise) through q3 (t near T, high noise).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)