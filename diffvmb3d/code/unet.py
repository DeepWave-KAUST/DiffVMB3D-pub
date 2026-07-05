"""
U-Net architecture for DiffVMB3D: Depth-progressive 3D velocity model building
via 2D generative diffusion models.

This module implements the 2D U-Net denoising network f_theta described in
Section III.III and illustrated in Figure 1 of the paper. The network operates
under the depth-as-channel formulation (Section III.I), where the depth dimension
nz of each 3D velocity patch is mapped to the channel dimension of standard 2D
convolutions, enabling volumetric VMB without 3D convolutions.

Architecture overview (Figure 1a):
    - 4 encoding + 4 decoding stages with skip connections
    - Feature channel dimensions at each stage: [C, 2C, 4C, 8C] where C = model_channels
      (e.g., [64, 128, 256, 512] with a 1024-channel bottleneck for C=64)
    - Downsampling: stride-2 3x3 convolutions
    - Upsampling: nearest-neighbor interpolation + 3x3 convolution
    - Attention blocks at deeper stages for long-range spatial dependencies

Multi-source conditioning injection (injected at every residual block):
    - Diffusion timestep t:    sinusoidal embedding -> MLP -> t_emb   (Figure 1b)
    - Depth position d_max:    sinusoidal embedding -> MLP -> d_emb   (Figure 1b)
    - Shallow velocity v_shallow: conv embedding -> c_emb             (Figure 1c)
    - Structural attribute s:  conv embedding -> s_emb                (Figure 1d)
    - Well velocity w:         conv embedding -> w_emb                (Figure 1f)
    - Well position l=(x,y):   positional encoding -> MLP -> l_emb   (Figure 1f)
"""

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)


class PositionalEncod(nn.Module):
    """
    Fourier positional encoding for 2D well lateral coordinates l = (x, y).

    Encodes each coordinate independently using multi-scale sinusoidal features:
        PE(l) = [l, sin(pi * 2^0 * x), ..., sin(pi * 2^{K-1} * x),
                     cos(pi * 2^0 * x), ..., cos(pi * 2^{K-1} * x),
                     sin(pi * 2^0 * y), ..., cos(pi * 2^{K-1} * y)]

    This encoding is used in the well location embedding layer (Figure 1f) to
    provide the network with the lateral position of the well within the patch,
    allowing it to spatially localize the well constraint.

    Args:
        PosEnc:  Number of frequency bands K per coordinate.
        device:  Device for the frequency tensors.

    Input:  [B, 2] tensor of (x, y) well coordinates.
    Output: [B, 2 + 4*K] tensor of positional features (raw coords + sin/cos for x and y).
    """

    def __init__(self, PosEnc=2, device='cuda'):
        super().__init__()
        self.PEnc = PosEnc
        # Frequency scales: pi * 2^k for k = 0, ..., K-1 (for x-coordinate)
        self.k_pi_sx = (th.tensor(np.pi) * (2 ** th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device)
        self.k_pi_sx = self.k_pi_sx.T

        # Frequency scales for y-coordinate (same structure)
        self.k_pi_sy = (th.tensor(np.pi) * (2 ** th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device)
        self.k_pi_sy = self.k_pi_sy.T

    def forward(self, input):
        # Compute sin/cos features for the x-coordinate
        tmpsx = th.cat([th.sin(self.k_pi_sx * input[:, 0]), th.cos(self.k_pi_sx * input[:, 0])], axis=0)
        # Compute sin/cos features for the y-coordinate
        tmpsy = th.cat([th.sin(self.k_pi_sy * input[:, 1]), th.cos(self.k_pi_sy * input[:, 1])], axis=0)
        # Concatenate raw coordinates with all positional features
        return th.cat([input, tmpsx.T, tmpsy.T], -1)


class PositionalEncod2(nn.Module):
    """
    Fourier positional encoding for a single 1D coordinate.

    Similar to PositionalEncod but encodes only one scalar input, used when
    only a single coordinate needs encoding (e.g., depth position in ablation).

    Args:
        PosEnc:  Number of frequency bands K.
        device:  Device for the frequency tensors.

    Input:  [B, 1] tensor of scalar coordinates.
    Output: [B, 1 + 2*K] tensor of positional features.
    """

    def __init__(self, PosEnc=2, device='cuda'):
        super().__init__()
        self.PEnc = PosEnc
        self.k_pi_sx = (th.tensor(np.pi) * (2 ** th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device)
        self.k_pi_sx = self.k_pi_sx.T

    def forward(self, input):
        tmpsx = th.cat([th.sin(self.k_pi_sx * input[:, 0]), th.cos(self.k_pi_sx * input[:, 0])], axis=0)
        return th.cat([input, tmpsx.T], -1)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (Figure 1b).

    Computes a positional embedding for scalar inputs (diffusion timestep t or
    depth coordinate d_max) using the standard sinusoidal encoding from
    Vaswani et al. (2017):
        emb_k = sin/cos(x * exp(-k * log(10000) / (dim/2)))

    In DiffVMB3D, this is used for both the diffusion timestep t and the
    depth position d_max, which are embedded independently before being
    concatenated and injected into each residual block (Figure 1b).

    Args:
        dim:    Embedding dimension (must be even).
        scale:  Linear scaling factor applied to the input before encoding.

    Input:  [B] tensor of scalar values (timestep or depth).
    Output: [B, dim] tensor of sinusoidal embeddings.
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        # Outer product: each input value x scaled and multiplied by each frequency
        emb = th.outer(x * self.scale, emb)
        # Concatenate sin and cos components to form the full embedding
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepBlock(nn.Module):
    """
    Abstract base class for modules that accept the full set of conditioning
    embeddings in their forward pass. All residual blocks in the U-Net inherit
    from this class.

    The forward signature passes all conditioning signals defined in the paper:
        time_emb:     Diffusion timestep embedding (t_emb)
        db_emb:       Depth position embedding (d_emb)
        cond_emb:     Shallow velocity embedding (c_emb)
        ref:          Structural attribute embedding (s_emb), or None
        well:         Well velocity embedding (w_emb), or None
        well_loc_emb: Well position embedding (l_emb), or None
    """

    @abstractmethod
    def forward(self, x, time_emb, db_emb, cond_emb, ref, well, well_loc_emb):
        """Apply the module to `x` given all conditioning embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential container that automatically routes conditioning embeddings
    to children that support them (TimestepBlock subclasses), while passing
    only the feature tensor to standard layers (e.g., Downsample, Upsample).
    """

    def forward(self, x, time_emb, db_emb, cond_emb, ref, well, well_loc_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb, db_emb, cond_emb, ref, well, well_loc_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Spatial upsampling by factor 2 using nearest-neighbor interpolation,
    optionally followed by a 3x3 convolution (as described in Section III.III).

    In the U-Net decoder, upsampling restores the spatial resolution at each
    stage before the skip connection from the corresponding encoder stage.

    Args:
        channels:   Number of input/output channels.
        use_conv:   If True, apply a learned 3x3 convolution after interpolation.
        dims:       Spatial dimensionality (2 for DiffVMB3D's 2D U-Net).
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            # For 3D signals, upsample only the inner two spatial dimensions.
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Spatial downsampling by factor 2 using a stride-2 convolution (as described
    in Section III.III) or average pooling.

    In the U-Net encoder, downsampling reduces the spatial resolution at each
    stage while the channel dimension increases.

    Args:
        channels:   Number of input/output channels.
        use_conv:   If True, use a learned stride-2 3x3 convolution (default
                    in DiffVMB3D); otherwise use average pooling.
        dims:       Spatial dimensionality (2 for DiffVMB3D's 2D U-Net).
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block with multi-source conditioning injection (Figure 1g).

    This is the core building block of the U-Net, responsible for integrating
    all conditioning signals into the feature representation at each resolution
    stage. The block performs:

    1. Main processing branch:
       - GroupNorm -> SiLU -> 3x3 Conv (in_layers)
       - Add concatenated time+depth embedding (t_emb || d_emb) after first conv
       - GroupNorm -> SiLU -> Dropout -> 3x3 Conv (out_layers)
       - Residual skip connection

    2. Conditioning branch (summed and concatenated with main branch):
       - Shallow velocity c_emb: adaptive pooling -> GN -> SiLU -> 3x3 Conv
       - Structural attribute s_emb: adaptive pooling -> GN -> SiLU -> 3x3 Conv
       - Well velocity w_emb: adaptive pooling -> GN -> SiLU -> 3x3 Conv,
         then add well position l_emb (projected through MLP)
       - All three are summed element-wise

    3. Fusion:
       - Concatenate main branch output and conditioning branch
       - GN -> SiLU -> 3x3 Conv to halve the channel dimension back

    This design allows each conditioning signal to influence the prediction at
    every spatial scale of the U-Net, as described in Section III.III.

    Args:
        channels:               Input feature channels.
        time_emb_channels:      Channels of the timestep embedding t_emb (and d_emb).
        dropout:                Dropout probability.
        out_channels:           Output feature channels (if None, same as input).
        cond_channels:          Channels of the shallow velocity embedding c_emb.
        ref_channels:           Channels of the structural attribute embedding s_emb.
        well_channels:          Channels of the well velocity embedding w_emb.
        wellloc_emb_dim:        Dimension of the well position embedding l_emb.
        use_conv:               If True, use 3x3 conv for the skip connection
                                when channels change; otherwise use 1x1 conv.
        use_scale_shift_norm:   If True, use scale-shift normalization instead
                                of additive conditioning.
        dims:                   Spatial dimensionality (2 for DiffVMB3D).
        use_checkpoint:         If True, use gradient checkpointing to reduce
                                memory at the cost of recomputation.
    """

    def __init__(
        self,
        channels,
        time_emb_channels,
        dropout,
        out_channels=None,
        cond_channels=None,
        ref_channels=None,
        well_channels=None,
        wellloc_emb_dim=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.time_emb_channels = time_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.cond_channels = cond_channels
        self.ref_channels = ref_channels
        self.well_channels = well_channels
        self.wellloc_emb_dim = wellloc_emb_dim
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # --- Main processing branch (Figure 1g, upper path) ---
        # First half: GN -> SiLU -> 3x3 Conv
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Time + depth embedding projection (Figure 1b -> Figure 1g).
        # The concatenated (t_emb || d_emb) vector of dimension 2*time_emb_channels
        # is projected and added to the feature map after the first convolution.
        self.time_emb_layers = nn.Sequential(
            SiLU(),
            linear(
                2 * time_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Second half: GN -> SiLU -> Dropout -> 3x3 Conv (zero-initialized)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # Residual skip connection: identity if channels match, otherwise
        # a 1x1 or 3x3 convolution to project to the output channel dimension.
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        # --- Conditioning branch: shallow velocity c_emb (Figure 1c -> 1g) ---
        # Spatially adapts c_emb to the current feature map resolution via
        # adaptive average pooling, then projects through GN -> SiLU -> 3x3 Conv.
        if cond_channels is not None:
            self.condtop_emb_conv = nn.Sequential(
                normalization(self.cond_channels),
                SiLU(),
                conv_nd(dims, self.cond_channels, self.out_channels, 3, padding=1),
            )

        # --- Conditioning branch: structural attribute s_emb (Figure 1d -> 1g) ---
        # Same spatial adaptation as c_emb. Set to None when s is dropped
        # via classifier-free guidance.
        if ref_channels is not None:
            self.ref_emb_conv = nn.Sequential(
                normalization(self.ref_channels),
                SiLU(),
                conv_nd(dims, self.ref_channels, self.out_channels, 3, padding=1),
            )

        # --- Conditioning branch: well velocity w_emb + position l_emb (Figure 1f -> 1g) ---
        if well_channels is not None:
            # Well velocity: spatial adaptation via adaptive pooling + GN -> SiLU -> 3x3 Conv
            self.well_emb_conv = nn.Sequential(
                normalization(self.well_channels),
                SiLU(),
                conv_nd(dims, self.well_channels, self.out_channels, 3, padding=1),
            )

            # Well location embedding projection: projects l_emb and adds it to
            # the well velocity feature map, informing the network of the lateral
            # position of the well within the current patch.
            self.wellloc_emb_layers = nn.Sequential(
                SiLU(),
                linear(
                    wellloc_emb_dim,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

            # Fusion layer for well velocity + well location: GN -> SiLU -> Dropout -> 3x3 Conv
            self.well_fuse = nn.Sequential(
                normalization(self.out_channels),
                SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        # --- Final fusion: concatenate main branch and conditioning branch ---
        # After summing all conditioning contributions (c_emb + s_emb + w_emb),
        # concatenate with the residual output and reduce channels via
        # GN -> SiLU -> 3x3 Conv, producing the final block output O_mid.
        if cond_channels is not None:
            self.allcond_fuse = nn.Sequential(
                normalization(2 * self.out_channels),
                SiLU(),
                conv_nd(dims, 2 * self.out_channels, self.out_channels, 3, padding=1),
            )

    def forward(self, x, time_emb, db_emb, cond_emb, ref_emb=None, well_emb=None, well_loc_emb=None):
        """
        Forward pass with gradient checkpointing support.

        Args:
            x:            Feature map I_mid from the previous layer. Shape: [B, C, H, W].
            time_emb:     Diffusion timestep embedding t_emb. Shape: [B, D_t].
            db_emb:       Depth position embedding d_emb. Shape: [B, D_t].
            cond_emb:     Shallow velocity embedding c_emb. Shape: [B, C_cond, H0, W0].
            ref_emb:      Structural attribute embedding s_emb, or None. Shape: [B, C_ref, H0, W0].
            well_emb:     Well velocity embedding w_emb, or None. Shape: [B, C_well, H0, W0].
            well_loc_emb: Well position embedding l_emb, or None. Shape: [B, D_l].

        Returns:
            Output feature map O_mid. Shape: [B, C_out, H, W].
        """
        return checkpoint(
            self._forward, (x, time_emb, db_emb, cond_emb, ref_emb, well_emb, well_loc_emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, time_emb, db_emb, cond_emb, ref_emb=None, well_emb=None, well_loc_emb=None):
        # === Main processing branch ===
        # Step 1: GN -> SiLU -> 3x3 Conv
        h = self.in_layers(x)

        # Step 2: Inject concatenated time + depth embedding (t_emb || d_emb).
        # This informs each residual block of both the current diffusion noise
        # level and the absolute depth position of the patch (Figure 1b).
        time_emb_out = self.time_emb_layers(th.cat((time_emb, db_emb), dim=-1)).type(h.dtype)

        # Broadcast the 1D embedding to match the spatial dimensions of h.
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]

        if self.use_scale_shift_norm:
            # Scale-shift conditioning: h = norm(h) * (1 + scale) + shift
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(time_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # Additive conditioning: h = h + time_emb_out
            h = h + time_emb_out
            # Step 3: GN -> SiLU -> Dropout -> 3x3 Conv
            h = self.out_layers(h)

        # Step 4: Residual connection (skip_connection projects x if channels differ)
        out = self.skip_connection(x) + h

        # === Conditioning branch ===
        B, C, H, W = out.shape

        # Shallow velocity c_emb: spatially adapt to current resolution via
        # adaptive average pooling, then project through GN -> SiLU -> Conv.
        cond_ds = F.adaptive_avg_pool2d(cond_emb, output_size=(H, W))
        cond_ds = self.condtop_emb_conv(cond_ds)

        # Well velocity w_emb + well position l_emb (when well data is available).
        if well_emb is not None:
            # Spatially adapt well embedding to current resolution.
            well_ds = F.adaptive_avg_pool2d(well_emb, output_size=(H, W))
            well_ds = self.well_emb_conv(well_ds)

            # Project well location embedding and inject into the well feature map.
            wellloc_emb_out = self.wellloc_emb_layers(well_loc_emb).type(h.dtype)
            while len(wellloc_emb_out.shape) < len(out.shape):
                wellloc_emb_out = wellloc_emb_out[..., None]

            if self.use_scale_shift_norm:
                out_norm, out_rest = self.well_fuse[0], self.well_fuse[1:]
                scale, shift = th.chunk(wellloc_emb_out, 2, dim=1)
                well_ds = out_norm(well_ds) * (1 + scale) + shift
                well_ds = out_rest(well_ds)
            else:
                well_ds = well_ds + wellloc_emb_out
                well_ds = self.well_fuse(well_ds)

            # Sum well contribution into the conditioning branch.
            cond_ds = cond_ds + well_ds

        # Structural attribute s_emb (when structural data is available).
        if ref_emb is not None:
            ref_ds = F.adaptive_avg_pool2d(ref_emb, output_size=(H, W))
            ref_ds = self.ref_emb_conv(ref_ds)

            # Sum structural contribution into the conditioning branch.
            cond_ds = cond_ds + ref_ds

        # === Final fusion: concatenate main + conditioning, then reduce channels ===
        # [B, 2*C_out, H, W] -> GN -> SiLU -> Conv -> [B, C_out, H, W]
        fused = th.cat([out, cond_ds], dim=1)
        out = self.allcond_fuse(fused)

        return out


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block (Figure 1e).

    Applied at the deeper stages of the U-Net to capture long-range spatial
    dependencies across the lateral plane (nx x ny). The block operates on
    the intermediate feature map O_mid from the preceding residual block.

    Architecture:
        1. GroupNorm on input
        2. Three parallel 1x1 convolutions produce Q, K, V
        3. Scaled dot-product attention: softmax(Q^T K / sqrt(d)) V
        4. 1x1 convolution on the attended output (zero-initialized)
        5. Residual connection: output = input + attended

    Args:
        channels:        Number of input/output feature channels.
        num_heads:       Number of attention heads (default 4).
        use_checkpoint:  If True, use gradient checkpointing.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        # Single 1x1 conv producing concatenated Q, K, V (3x channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()

        # Zero-initialized output projection for stable training initialization
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        # Flatten spatial dimensions for 1D attention computation
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        # Split into multi-head format: [B*num_heads, C/num_heads * 3, N]
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        # Residual connection and restore spatial shape
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Scaled dot-product QKV attention module.

    Computes attention weights via:
        weight = softmax(Q * K^T / sqrt(d))
        output = weight * V

    The scaling by 1/sqrt(sqrt(d)) on both Q and K (instead of 1/sqrt(d) on
    the product) improves numerical stability in float16.
    """

    def forward(self, qkv):
        """
        Args:
            qkv:  [B*num_heads, 3*d_head, N] tensor of concatenated Q, K, V.

        Returns:
            [B*num_heads, d_head, N] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        # Split scaling between Q and K for fp16 stability
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """FLOPs counter for the `thop` profiling package."""
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # Two matrix multiplications: QK^T and (softmax * V)
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class CrossEfficientAttention(nn.Module):
    """
    Cross-attention with linear complexity using efficient attention
    (Shen et al., 2021).

    Instead of standard softmax attention (O(N^2)), this uses row-wise and
    column-wise softmax on Q and K respectively, reducing complexity to O(N*C).
    Used for cross-attention between the feature map and conditioning embeddings.

    Args:
        dims:            Spatial dimensionality.
        channels:        Number of input/output channels.
        num_heads:       Number of attention heads.
        use_checkpoint:  If True, use gradient checkpointing.
    """

    def __init__(self, dims, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.dims = dims
        self.channels = channels
        self.num_heads = num_heads
        self.dim_head = channels // num_heads
        self.scale = 1 / math.sqrt(math.sqrt(channels))
        self.use_checkpoint = use_checkpoint

        self.to_q = conv_nd(dims, channels, channels, 1)
        self.to_kv = conv_nd(dims, channels, 2 * channels, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, x, cproj):
        """
        Args:
            x:     Query feature map [B, C, H, W].
            cproj: Conditioning feature map [B, C, H, W] (key/value source).
        """
        return checkpoint(self._forward, (x, cproj), self.parameters(), self.use_checkpoint)

    def _forward(self, x, cproj):
        B, C, H, W = x.shape
        N = H * W

        Q = self.to_q(x)               # [B, C, H, W]
        KV = self.to_kv(cproj)         # [B, 2C, H, W]

        Q = Q.reshape(B, C, N)         # [B, C, N]
        KV = KV.reshape(B, 2 * C, N)   # [B, 2C, N]
        K, V = KV.split(C, dim=1)      # Each [B, C, N]

        # Efficient attention: row-softmax on Q, column-softmax on K
        q = F.softmax(Q * self.scale, dim=1)   # Normalize over channels
        k = F.softmax(K, dim=2)                # Normalize over spatial positions

        # Linear-complexity attention: context = K @ V^T, out = context @ Q
        context = th.bmm(k, V.transpose(1, 2))  # [B, C, C]
        out = th.bmm(context, q)                 # [B, C, N]

        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return out


class SelfEfficientAttention(nn.Module):
    """
    Self-attention with linear complexity using efficient attention.

    Same mechanism as CrossEfficientAttention but Q, K, V are all derived
    from the same input feature map (self-attention).

    Args:
        channels:        Number of input/output channels.
        num_heads:       Number of attention heads.
        use_checkpoint:  If True, use gradient checkpointing.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.dim_head = channels // num_heads
        self.scale = 1 / math.sqrt(math.sqrt(channels))
        self.use_checkpoint = use_checkpoint

        self.to_qkv = conv_nd(1, channels, 3 * channels, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        B, C, *spatial = x.shape
        H, W = spatial[-2], spatial[-1]
        N = H * W

        qkv = self.to_qkv(x)                     # [B, 3C, H, W]
        q, k, v = th.split(qkv, C, dim=1)         # Each [B, C, H, W]

        q = q.view(B, self.num_heads, self.dim_head, N)  # [B, h, d_h, N]
        k = k.view(B, self.num_heads, self.dim_head, N)
        v = v.view(B, self.num_heads, self.dim_head, N)

        # Efficient attention with multi-head decomposition
        q = F.softmax(q * self.scale, dim=2)  # Row-softmax over d_h
        k = F.softmax(k, dim=3)               # Column-softmax over N

        # Linear-complexity: context = K @ V^T -> [B, h, d_h, d_h]
        context = th.einsum('b h d n, b h e n -> b h d e', k, v)
        # out = context @ Q -> [B, h, d_h, N]
        out = th.einsum('b h d e, b h d n -> b h e n', context, q)

        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        # Residual connection
        return x + out


class LinearAttention(nn.Module):
    """
    Memory-efficient linearized attention using the ELU+1 feature map
    to approximate softmax attention with O(N) complexity.

    Uses phi(x) = elu(x) + 1 as the kernel feature map:
        Attention(Q, K, V) ≈ phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ phi(K)^T @ 1)
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: [B, N, C] tensors of queries, keys, and values.

        Returns:
            [B, N, C] tensor after linear attention.
        """
        phi = lambda x: F.elu(x) + 1
        Q_phi = phi(Q)  # [B, N, C]
        K_phi = phi(K)  # [B, N, C]
        # Compute KV context: [B, C, C]
        KV = th.einsum('bnc,bnd->bcd', K_phi, V)
        # Normalization denominator
        denom = th.einsum('bnc,bnc->bn', Q_phi, K_phi.sum(dim=1, keepdim=True).expand_as(Q_phi))
        Z = 1.0 / (denom + self.eps)
        # Attend: [B, N, C]
        out = th.einsum('bnc,bcd->bnd', Q_phi, KV)
        # Apply normalization
        out = out * Z.unsqueeze(-1)
        return out


class UNetModel(nn.Module):
    """
    The complete 2D U-Net denoising network f_theta for DiffVMB3D (Figure 1a).

    This network operates under the depth-as-channel formulation (Section III.I):
    each 3D velocity patch of size [nz, ny, nx] is treated as a 2D feature map
    of size [ny, nx] with nz channels. The network takes as input the concatenation
    of the noised deep velocity patch x_t and the shallow velocity embedding c_emb,
    and predicts the clean velocity patch x_0 (= v_deep) under the x0-prediction
    parameterization (Eq. 3).

    Conditioning signals are injected as follows:
        - v_shallow:  Embedded by cond_embed (2x [Conv3x3 + GN + SiLU]), then
                      (1) concatenated with x_t as network input, and
                      (2) injected into every residual block via adaptive pooling
        - t:          Sinusoidal embedding + MLP (time_embed), injected into
                      every residual block
        - d_max:      Sinusoidal embedding + MLP (db_embed), concatenated with
                      t_emb and injected into every residual block
        - s:          Embedded by ref_embed (2x [Conv3x3 + GN + SiLU]), injected
                      into every residual block; set to None when dropped by CFG
        - w:          Embedded by well_embed (2x [Conv3x3 + GN + SiLU]), injected
                      into every residual block; set to None when dropped by CFG
        - l=(x,y):    Fourier positional encoding + MLP (wellloc_embed), injected
                      into every residual block alongside w

    Args:
        in_channels:             Input channels (= nz, depth samples per patch;
                                 16 in the paper).
        model_channels:          Base channel count C (64 in the paper).
        out_channels:            Output channels (= nz = in_channels).
        num_res_blocks:          Number of residual blocks per encoder/decoder stage.
        attention_resolutions:   Set of downsampling rates at which attention is applied.
        dropout:                 Dropout probability.
        channel_mult:            Channel multiplier per stage (e.g., (1,2,4,8) gives
                                 channels [64, 128, 256, 512]).
        conv_resample:           If True, use learned convolutions for up/downsampling.
        dims:                    Spatial dimensionality (2 for DiffVMB3D).
        time_emb_scale:          Scale factor for the sinusoidal timestep encoding.
        num_classes:             Number of classes for class-conditional generation
                                 (None for DiffVMB3D).
        use_checkpoint:          If True, use gradient checkpointing.
        num_heads:               Number of attention heads.
        num_heads_upsample:      Number of attention heads in the decoder (defaults
                                 to num_heads).
        use_scale_shift_norm:    If True, use scale-shift normalization.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        time_emb_scale=1.0,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # Padding size to ensure spatial dimensions are divisible by 2^num_stages.
        self.padder_size = 2 ** len(channel_mult)

        # ===== Embedding layers (Figure 1b-f) =====

        # Diffusion timestep embedding (Figure 1b, left branch):
        # t -> sinusoidal encoding -> Linear -> SiLU -> Linear -> t_emb
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Shallow velocity input embedding (Figure 1c):
        # v_shallow -> [Conv3x3 + GN + SiLU] x 2 -> c_emb
        # Projects from nz (= in_channels) depth channels to model_channels.
        # c_emb serves dual roles: (1) concatenated with x_t as network input,
        # and (2) injected into every residual block.
        self.cond_embed = nn.Sequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        # Depth position embedding (Figure 1b, right branch):
        # d_max -> sinusoidal encoding -> Linear -> SiLU -> Linear -> d_emb
        # Embedded analogously to the timestep t; the two are concatenated
        # before injection into residual blocks.
        self.db_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Structural attribute embedding (Figure 1d):
        # s -> [Conv3x3 + GN + SiLU] x 2 -> s_emb
        # Projects the structural image (e.g., convolution image) from nz channels
        # to model_channels. Set to None when dropped by classifier-free guidance.
        self.ref_embed = nn.Sequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        # Well velocity embedding (Figure 1f, upper branch):
        # w -> [Conv3x3 + GN + SiLU] x 2 -> w_emb
        # The 1D well velocity profile, expanded to patch dimensions along
        # the lateral directions, is projected to model_channels.
        self.well_embed = nn.Sequential(
            conv_nd(dims, in_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        # Well location embedding (Figure 1f, lower branch):
        # l=(x,y) -> Fourier positional encoding -> Linear -> SiLU -> Linear -> l_emb
        # Encodes the 2D lateral position of the well within the patch.
        wellloc_embed_dim = model_channels * 4
        self.wellloc_embed = nn.Sequential(
            PositionalEncod(model_channels // 2),
            linear(2 * (model_channels + 1), wellloc_embed_dim),
            SiLU(),
            linear(wellloc_embed_dim, wellloc_embed_dim),
        )

        # Optional class embedding (not used in DiffVMB3D).
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ===== Initial convolution =====
        # Concatenation of x_t (in_channels = nz) and c_emb (model_channels)
        # is projected to model_channels via a 3x3 convolution.
        self.inp = conv_nd(dims, in_channels + model_channels, model_channels, 3, padding=1)

        # ===== Encoder (downsampling path) =====
        # Each stage contains num_res_blocks residual blocks, optionally followed
        # by an attention block. Downsampling between stages uses stride-2 convolutions.
        self.downs = nn.ModuleList([])
        encoder_channels = [model_channels]  # Track channel dims for skip connections
        ch = model_channels
        ds = 1  # Current downsampling factor
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        cond_channels=model_channels,
                        ref_channels=model_channels,
                        well_channels=model_channels,
                        wellloc_emb_dim=wellloc_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # Add attention block at specified downsampling rates
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.downs.append(TimestepEmbedSequential(*layers))
                encoder_channels.append(ch)
            # Add downsampling layer between stages (except the last)
            if level != len(channel_mult) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                encoder_channels.append(ch)
                ds *= 2

        # ===== Bottleneck (middle block) =====
        # ResBlock -> AttentionBlock -> ResBlock at the deepest resolution.
        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                ref_channels=model_channels,
                well_channels=model_channels,
                wellloc_emb_dim=wellloc_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                ref_channels=model_channels,
                well_channels=model_channels,
                wellloc_emb_dim=wellloc_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ===== Decoder (upsampling path) =====
        # Mirror of the encoder with skip connections from corresponding encoder
        # stages. Each stage concatenates the encoder features before processing.
        self.ups = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + encoder_channels.pop(),  # Concatenated skip connection
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        cond_channels=model_channels,
                        ref_channels=model_channels,
                        well_channels=model_channels,
                        wellloc_emb_dim=wellloc_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                # Add upsampling at the end of each stage (except the first)
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(*layers))

        # ===== Output projection =====
        # GN -> SiLU -> 3x3 Conv (zero-initialized) projecting back to
        # out_channels (= nz), producing the predicted v_deep under the
        # depth-as-channel formulation.
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """Convert the encoder, bottleneck, and decoder to float16."""
        self.downs.apply(convert_module_to_f16)
        self.middle.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the encoder, bottleneck, and decoder back to float32."""
        self.downs.apply(convert_module_to_f32)
        self.middle.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """Get the dtype used by the encoder/decoder (fp16 or fp32)."""
        return next(self.downs.parameters()).dtype

    def forward(self, inp, cond_top, ref, db, well, well_loc, timesteps, y=None):
        """
        Forward pass of the conditional denoising U-Net f_theta.

        Implements the network prediction described in Section III.III:
            pred_x0 = f_theta(x_t, v_shallow, s, d_max, w, l, t)

        Args:
            inp:        Noised deep velocity patch x_t under the depth-as-channel
                        formulation. Shape: [B, nz, ny, nx].
            cond_top:   Shallow velocity patch v_shallow. Shape: [B, nz, ny, nx].
            ref:        Structural attribute s, or None if dropped by CFG.
                        Shape: [B, nz, ny, nx].
            db:         Depth position scalar d_max. Shape: [B, 1].
            well:       Well velocity w (expanded to patch dimensions), or None.
                        Shape: [B, nz, ny, nx].
            well_loc:   Well lateral position l = (x, y), or None. Shape: [B, 2].
            timesteps:  Diffusion timestep t. Shape: [B].
            y:          Class labels (unused in DiffVMB3D). Shape: [B].

        Returns:
            Predicted clean velocity patch v_deep (= x_0) under the
            depth-as-channel formulation. Shape: [B, nz, ny, nx].
        """
        b, c, h, w = inp.shape
        # Pad spatial dimensions to be divisible by 2^num_stages for safe
        # downsampling/upsampling through the encoder-decoder.
        inp = self.check_image_size(inp)
        cond_top = self.check_image_size(cond_top)

        # === Compute all conditioning embeddings ===

        # Shallow velocity embedding c_emb (Figure 1c):
        # v_shallow [B, nz, ny, nx] -> c_emb [B, model_channels, ny, nx]
        cond_top = self.cond_embed(cond_top)
        # Concatenate x_t and c_emb along channel dimension for network input
        x = th.cat([inp, cond_top], dim=1)

        # Depth position embedding d_emb (Figure 1b):
        # d_max [B] -> sinusoidal -> MLP -> d_emb [B, 4*model_channels]
        db_emb = self.db_embed(db.squeeze(1))

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Diffusion timestep embedding t_emb (Figure 1b):
        # t [B] -> sinusoidal -> MLP -> t_emb [B, 4*model_channels]
        time_emb = self.time_embed(timesteps)

        # Well velocity embedding w_emb and well location embedding l_emb (Figure 1f):
        # Only computed when well data is available (not dropped by CFG).
        if well is not None:
            # w [B, nz, ny, nx] -> w_emb [B, model_channels, ny, nx]
            well = self.well_embed(well)
            # l [B, 2] -> Fourier PE -> MLP -> l_emb [B, 4*model_channels]
            well_loc = self.wellloc_embed(well_loc)

        # Structural attribute embedding s_emb (Figure 1d):
        # Only computed when structural data is available (not dropped by CFG).
        if ref is not None:
            # s [B, nz, ny, nx] -> s_emb [B, model_channels, ny, nx]
            ref = self.ref_embed(ref)

        # Optional class-conditional embedding (unused in DiffVMB3D).
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            time_emb = time_emb + self.label_emb(y)

        # === U-Net forward pass ===

        # Initial projection: [B, nz + model_channels, ny, nx] -> [B, model_channels, ny, nx]
        skips = []
        x = x.type(self.inner_dtype)
        x = self.inp(x)
        skips.append(x)

        # Encoder: progressively downsample while injecting all conditioning.
        for module in self.downs:
            x = module(x, time_emb, db_emb, cond_top, ref, well, well_loc)
            skips.append(x)

        # Bottleneck: ResBlock -> Attention -> ResBlock at the coarsest resolution.
        x = self.middle(x, time_emb, db_emb, cond_top, ref, well, well_loc)

        # Decoder: progressively upsample with skip connections from the encoder.
        for module in self.ups:
            cat_in = th.cat([x, skips.pop()], dim=1)
            x = module(cat_in, time_emb, db_emb, cond_top, ref, well, well_loc)

        # Output projection: [B, model_channels, ny, nx] -> [B, nz, ny, nx]
        x = x.type(inp.dtype)
        x = self.out(x)
        # Remove padding to restore the original spatial dimensions.
        return x[:, :, :h, :w]

    def check_image_size(self, x):
        """
        Pad the input tensor so that its spatial dimensions (H, W) are
        divisible by 2^num_stages (= padder_size). This prevents dimension
        mismatches during downsampling and upsampling in the U-Net.

        Uses replicate padding to avoid boundary artifacts.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x