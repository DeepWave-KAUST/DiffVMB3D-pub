"""
Configuration and factory utilities for DiffVMB3D model and diffusion setup.

This module serves as the central entry point for constructing the U-Net
denoising network f_theta and the Gaussian diffusion process with the specific
hyperparameters used in the DiffVMB3D framework. It translates high-level
configuration flags (e.g., predict_xstart, noise_schedule, timestep_respacing)
into the concrete model and diffusion objects.

Default DiffVMB3D configuration (from Section IV of the paper):
    - U-Net: in_channels=16 (nz=16 depth samples per patch), model_channels=64,
      channel_mult=(1,2,4,8,16) -> feature channels [64, 128, 256, 512, 1024],
      num_res_blocks=2, attention at 16x and 32x downsampling, scale-shift norm.
    - Diffusion: T=1000 steps, cosine beta schedule, x0-prediction
      (predict_xstart=True), fixed small variance, MSE loss.
    - Inference: "ddim10" timestep respacing for 10-step DDIM sampling.
"""

import argparse
import inspect
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Return the default hyperparameter dictionary for DiffVMB3D.

    These defaults correspond to the configuration described in Section IV
    of the paper (Training Setup):
        - in/out_channels = 16:     nz = 16 depth samples per velocity patch
        - num_channels = 64:        Base channel count C of the U-Net
        - channel_mult = (1,2,4,8,16): 5 encoder/decoder stages with channels
                                    [64, 128, 256, 512, 1024]
        - num_res_blocks = 2:       Two residual blocks per stage
        - attention_resolutions = (16, 32): Attention at the two deepest stages
        - noise_schedule = "cosine": Cosine variance schedule
        - diffusion_steps = 1000:   Total diffusion timesteps T
        - predict_xstart = True:    x0-prediction parameterization (Eq. 3)
        - rescale_timesteps = True: Scale timesteps to [0, 1000] for the U-Net
        - use_scale_shift_norm = True: Scale-shift conditioning in ResBlocks
        - use_checkpoint = False:   Gradient checkpointing (enable to save memory)

    Returns:
        Dict of hyperparameter name -> default value.
    """
    return dict(
        in_channels=16,                     # nz: depth samples per patch
        num_channels=64,                    # Base channel count C
        out_channels=16,                    # Output channels (= in_channels = nz)
        channel_mult=(1, 2, 4, 8, 16),      # Channel multipliers per U-Net stage
        num_res_blocks=2,                   # Residual blocks per stage
        num_heads=4,                        # Attention heads
        num_heads_upsample=-1,              # Decoder attention heads (-1 = same as encoder)
        attention_resolutions=(16, 32),     # Downsampling rates for attention
        dropout=0.0,                        # Dropout probability
        learn_sigma=False,                  # If True, learn the variance (not used)
        sigma_small=False,                  # If True, use posterior variance (FIXED_SMALL)
        class_cond=False,                   # Class-conditional generation (not used)
        diffusion_steps=1000,               # Total diffusion timesteps T
        noise_schedule="cosine",            # Beta schedule type
        timestep_respacing="",              # Respacing string (e.g., "ddim10" for inference)
        use_kl=False,                       # If True, use KL loss instead of MSE
        predict_xstart=True,                # x0-prediction (Eq. 3); False = epsilon-prediction
        rescale_timesteps=True,             # Rescale timesteps to [0, 1000]
        rescale_learned_sigmas=False,       # Rescale learned sigmas (not used)
        use_checkpoint=False,               # Gradient checkpointing for memory savings
        use_scale_shift_norm=True,          # Scale-shift normalization in ResBlocks
    )


def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    use_wellguide,
):
    """
    Create and return both the U-Net model and the diffusion process.

    This is the main factory function called by training and inference scripts.
    It delegates to create_model() for the U-Net architecture and
    create_gaussian_diffusion() for the diffusion process setup.

    Args:
        All hyperparameters from model_and_diffusion_defaults(), plus:
        use_wellguide:  If True, enable optional well-log gradient guidance
                        during sampling.

    Returns:
        Tuple of (model, diffusion):
            model:     UNetModel instance (the denoising network f_theta).
            diffusion: SpacedDiffusion instance (handles training loss and sampling).
    """
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        use_wellguide=use_wellguide,
    )
    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Instantiate the 2D U-Net denoising network f_theta (Figure 1a).

    Constructs a UNetModel with the depth-as-channel formulation:
    in_channels = nz (depth samples per patch), and the lateral plane (ny x nx)
    is treated as the 2D spatial domain for standard 2D convolutions.

    Args:
        in_channels:             Input channels (= nz = 16 in the paper).
        num_channels:            Base channel count C (= 64 in the paper).
        out_channels:            Output channels (= nz = 16).
        channel_mult:            Per-stage channel multipliers.
        num_res_blocks:          Residual blocks per encoder/decoder stage.
        learn_sigma:             If True, double the output channels to also
                                 predict variance (not used in DiffVMB3D).
        class_cond:              If True, enable class-conditional generation.
        use_checkpoint:          If True, use gradient checkpointing.
        attention_resolutions:   Downsampling rates at which to apply attention.
        num_heads:               Number of attention heads in encoder.
        num_heads_upsample:      Number of attention heads in decoder.
        use_scale_shift_norm:    If True, use scale-shift normalization.
        dropout:                 Dropout probability.

    Returns:
        UNetModel instance.
    """
    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    use_wellguide=False,
):
    """
    Instantiate the Gaussian diffusion process with optional timestep respacing.

    This function configures the complete diffusion pipeline:
      1. Generates the beta schedule {beta_t}_{t=1}^{T} (cosine in DiffVMB3D).
      2. Selects the loss type (MSE for DiffVMB3D).
      3. Selects the model parameterization (x0-prediction for DiffVMB3D).
      4. Selects the variance type (FIXED_LARGE by default, FIXED_SMALL if
         sigma_small=True).
      5. Applies timestep respacing if specified (e.g., "ddim10" for 10-step
         DDIM inference).

    For DiffVMB3D, the default configuration is:
        steps=1000, noise_schedule="cosine", predict_xstart=True,
        loss_type=MSE, model_var_type=FIXED_LARGE, rescale_timesteps=True.

    At inference time, timestep_respacing="ddim10" reduces the schedule to
    10 steps via SpacedDiffusion.

    Args:
        steps:                   Total diffusion timesteps T (1000 in the paper).
        learn_sigma:             If True, model predicts variance (LEARNED_RANGE).
        sigma_small:             If True, use posterior variance (FIXED_SMALL)
                                 instead of beta (FIXED_LARGE).
        noise_schedule:          Beta schedule name ("cosine" for DiffVMB3D).
        use_kl:                  If True, use RESCALED_KL loss.
        predict_xstart:          If True, use x0-prediction (ModelMeanType.START_X);
                                 otherwise use epsilon-prediction.
        rescale_timesteps:       If True, rescale timesteps to [0, 1000].
        rescale_learned_sigmas:  If True, use RESCALED_MSE loss.
        timestep_respacing:      String specifying step reduction (e.g., "ddim10",
                                 "10,10,10", or "" for no respacing).
        use_wellguide:           If True, enable well-log gradient guidance.

    Returns:
        SpacedDiffusion instance ready for training or sampling.
    """
    # Generate the full T-step beta schedule.
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Select the training loss type.
    if use_kl:
        # Variational lower-bound loss (not used in DiffVMB3D).
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        # MSE with rescaled VLB for learned variance (not used in DiffVMB3D).
        loss_type = gd.LossType.RESCALED_MSE
    else:
        # Standard MSE loss (default for DiffVMB3D, Eq. 3-4).
        loss_type = gd.LossType.MSE

    # If no respacing is specified, use all T timesteps (training mode).
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        # Select the subset of timesteps (all T for training, 10 for DDIM inference).
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        # Model parameterization: x0-prediction (START_X) or epsilon-prediction.
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # Variance type: FIXED_LARGE (beta_t) or FIXED_SMALL (posterior variance),
        # or LEARNED_RANGE if learn_sigma is True.
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        use_wellguide=use_wellguide,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add all entries from a default hyperparameter dictionary to an
    argparse.ArgumentParser, automatically inferring the type of each
    argument from the default value. Boolean values use str2bool for
    flexible parsing (e.g., "yes", "true", "1" all map to True).

    Args:
        parser:        An argparse.ArgumentParser instance.
        default_dict:  Dict of parameter name -> default value.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of attributes from a parsed argparse.Namespace into a dict.

    Args:
        args:  Parsed argparse.Namespace object.
        keys:  Iterable of attribute names to extract.

    Returns:
        Dict of {key: value} for each key in keys.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Parse a boolean value from a string, supporting multiple common formats.
    Used as the type converter for boolean argparse arguments.

    Accepted true values:  "yes", "true", "t", "y", "1"
    Accepted false values: "no", "false", "f", "n", "0"
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")