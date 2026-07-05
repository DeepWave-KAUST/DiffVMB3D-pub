"""
Training entry point for DiffVMB3D: Depth-progressive 3D velocity model
building via 2D generative diffusion models.

This script orchestrates the complete training pipeline:
  1. Parse hyperparameters from command-line arguments
  2. Construct the 2D U-Net (f_theta) and Gaussian diffusion process
  3. Initialize the data loader for (v_deep, v_shallow, s, d_max, w, l) tuples
  4. Launch the training loop with classifier-free guidance dropout,
     EMA tracking, and periodic checkpointing

Default configuration matches Section IV of the paper:
  - AdamW optimizer with lr=1e-4, batch_size=24
  - EMA decay rate 0.999
  - Classifier-free guidance dropout: 5% for both well (w) and structural (s)
  - Cosine beta schedule, T=1000, x0-prediction, MSE loss
  - Single A100 GPU, ~200K iterations (~36 hours)

Usage:
    python train.py --data_dir ../dataset/train/ --batch_size 24 --lr 1e-4
"""

import argparse
from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import torch as th


def main():
    # Parse command-line arguments (model architecture + diffusion config + training config).
    args = create_argparser().parse_args()
    logger.configure()

    # Set compute device to GPU.
    device = th.device('cuda')

    # === Step 1: Create model and diffusion ===
    logger.log("creating model and diffusion...")
    # Extract only the model/diffusion-related hyperparameters from args.
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    # Instantiate the U-Net f_theta and the SpacedDiffusion process.
    # For training, all T=1000 timesteps are used (no respacing).
    model, diffusion = create_model_and_diffusion(
        **params,
        use_wellguide=args.use_wellguide,
    )

    # Optional: load pretrained weights for fine-tuning or transfer learning.
    # pretrained_dict = th.load('pretrained_model.pt', map_location=device)
    # model.load_state_dict(pretrained_dict)

    model.to(device)

    # === Step 2: Create timestep sampler ===
    # Default is UniformSampler: t ~ Uniform{0, ..., T-1} at each iteration.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # === Step 3: Create data loader ===
    logger.log("creating data loader...")
    # The data loader yields training tuples:
    #   (v_deep, v_shallow, s, d_max, w, l, cond)
    # where depth_size = out_channels = nz (16) determines the depth dimension
    # of each velocity patch under the depth-as-channel formulation.
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        depth_size=args.out_channels,  # nz: depth samples per patch (16)
        device=device,
        class_cond=args.class_cond,
    )

    # === Step 4: Launch training loop ===
    logger.log("training...")
    # The TrainLoop handles the complete training procedure:
    #   - Forward diffusion (noising v_deep to x_t) and x0-prediction loss (Eq. 3-4)
    #   - Classifier-free guidance dropout: independently drop well (w, l) with
    #     probability wellcond_drop=5% and structural attribute (s) with
    #     probability refcond_drop=5% (Section II)
    #   - AdamW optimizer with optional learning rate annealing
    #   - EMA weight tracking (decay=0.999) for stable inference sampling
    #   - Periodic checkpoint saving (every save_interval steps)
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        wellcond_drop=args.wellcond_drop,
        refcond_drop=args.refcond_drop,
    ).run_loop()


def create_argparser():
    """
    Build the argument parser with all training hyperparameters.

    Training-specific defaults (not part of model_and_diffusion_defaults):
        data_dir:           Path to the training dataset directory.
        schedule_sampler:   Timestep sampling strategy ("uniform" for DiffVMB3D).
        lr:                 Learning rate (1e-4 in the paper).
        weight_decay:       L2 regularization coefficient for AdamW.
        lr_anneal_steps:    If > 0, linearly anneal lr to zero over this many steps.
        batch_size:         Training batch size (24 in the paper).
        microbatch:         Sub-batch size for gradient accumulation (-1 = disabled).
        ema_rate:           EMA decay rate(s) for model weights (0.999 in the paper).
        log_interval:       Steps between logging loss/grad statistics.
        save_interval:      Steps between saving EMA checkpoints.
        resume_checkpoint:  Path to resume training from a saved checkpoint.
        use_fp16:           Enable mixed-precision (float16) training.
        fp16_scale_growth:  Loss scale growth rate per step in fp16 mode.
        wellcond_drop:      CFG dropout probability for well constraint (5%).
        refcond_drop:       CFG dropout probability for structural attribute (5%).
        use_wellguide:      Enable well-log gradient guidance during sampling.

    These are merged with model_and_diffusion_defaults() so that all
    architecture and diffusion hyperparameters are also configurable from
    the command line.

    Returns:
        argparse.ArgumentParser with all registered arguments.
    """
    defaults = dict(
        data_dir="../dataset/train/",
        schedule_sampler="uniform",         # Uniform timestep sampling
        lr=1e-4,                            # AdamW learning rate
        weight_decay=0.0,                   # L2 regularization
        lr_anneal_steps=0,                  # 0 = no annealing (constant lr)
        batch_size=24,                      # Batch size (24 in the paper)
        microbatch=-1,                      # -1 disables gradient accumulation
        ema_rate="0.999",                   # EMA decay rate for stable sampling
        log_interval=100,                   # Log every 100 steps
        save_interval=10000,                # Save checkpoint every 10K steps
        resume_checkpoint="",               # Empty = train from scratch
        use_fp16=False,                     # Mixed-precision training
        fp16_scale_growth=1e-3,             # Loss scale growth for fp16
        wellcond_drop=0.05,                 # 5% CFG dropout for well (w, l)
        refcond_drop=0.05,                  # 5% CFG dropout for structural (s)
        use_wellguide=False,                # Well gradient guidance (inference only)
    )
    # Merge with model/diffusion defaults so all hyperparameters are CLI-configurable.
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()