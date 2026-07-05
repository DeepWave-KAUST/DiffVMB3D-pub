"""
Diffusion timestep sampling strategies for DiffVMB3D training.

During training, each iteration requires sampling a diffusion timestep t from
{0, 1, ..., T-1} for each sample in the batch. The choice of sampling distribution
affects training efficiency and convergence.

This module provides two strategies:
  - UniformSampler:  Samples t uniformly from {0, ..., T-1}. This is the default
                     sampler used in DiffVMB3D, where each timestep is equally
                     likely and the importance weight is 1.0 for all timesteps.
  - LossSecondMomentResampler:  An adaptive sampler that preferentially samples
                     timesteps with higher loss variance (measured by the second
                     moment of recent losses). This focuses training effort on
                     timesteps where the model struggles most, but requires a
                     warm-up period to collect loss statistics.

Both samplers use importance sampling to maintain an unbiased training objective:
the returned weights compensate for the non-uniform sampling probability so that
the expected gradient remains identical to uniform sampling.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Factory function to create a timestep sampler by name.

    Args:
        name:       Sampler name ("uniform" or "loss-second-moment").
        diffusion:  The GaussianDiffusion object (provides num_timesteps = T).

    Returns:
        A ScheduleSampler instance.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for diffusion timestep samplers.

    Subclasses define a probability distribution over timesteps {0, ..., T-1}
    via the weights() method. The sample() method performs importance sampling:
    timesteps are drawn according to the normalized weights, and each sampled
    timestep is accompanied by an importance weight that corrects for the
    non-uniform sampling probability, keeping the training objective unbiased:

        importance_weight[i] = 1 / (T * p[i])

    where p[i] = w[i] / sum(w) is the normalized sampling probability for
    timestep i. For UniformSampler, p[i] = 1/T and all importance weights are 1.
    """

    @abstractmethod
    def weights(self):
        """
        Return a 1-D numpy array of non-negative weights, one per diffusion
        timestep. The weights need not be normalized but must be positive.
        Higher weights increase the probability of sampling that timestep.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample a batch of diffusion timesteps.

        Draws timestep indices from the categorical distribution defined by
        self.weights(), and computes the corresponding importance weights to
        keep the training loss expectation unbiased.

        Args:
            batch_size:  Number of timesteps to sample (= batch size B).
            device:      Torch device for the output tensors.

        Returns:
            Tuple of:
              - indices:  [B] integer tensor of sampled timestep indices.
              - weights:  [B] float tensor of importance weights for loss scaling.
        """
        w = self.weights()
        # Normalize weights to a valid probability distribution.
        p = w / np.sum(w)
        # Draw timestep indices according to the probability distribution.
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        # Importance weights: 1 / (T * p[i]) to correct for non-uniform sampling.
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniform timestep sampler (default for DiffVMB3D).

    Assigns equal weight to all T timesteps, so each timestep is sampled with
    probability 1/T and the importance weight is identically 1.0. This is the
    standard choice used in most DDPM implementations and in DiffVMB3D training.
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        # All weights equal to 1 -> uniform distribution over {0, ..., T-1}.
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Abstract base class for samplers that adapt their timestep distribution
    based on observed training losses.

    This class handles the distributed synchronization needed when training
    across multiple GPUs: each rank reports its local losses, and all ranks
    synchronize to maintain identical sampling distributions.

    Subclasses must implement update_with_all_losses() to define how the
    loss history is used to update the sampling weights.
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Synchronize per-timestep losses across all distributed ranks, then
        update the sampling weights.

        This method is called from each rank after computing the training loss
        for a batch. It performs an all-gather to collect timestep-loss pairs
        from all ranks, then delegates to update_with_all_losses() which
        deterministically updates the internal state identically on all ranks.

        Args:
            local_ts:     [B] integer tensor of timesteps used in this batch.
            local_losses: [B] float tensor of per-sample losses (detached).
        """
        # Gather batch sizes from all ranks to handle potentially different sizes.
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad tensors to the maximum batch size for all_gather compatibility.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Flatten gathered results, trimming padding from each rank.
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update internal sampling weights using synchronized losses from all ranks.

        Must be deterministic to ensure all ranks maintain identical state.

        Args:
            ts:     List of integer timesteps from all ranks.
            losses: List of float losses corresponding to each timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Adaptive timestep sampler that weights timesteps by the second moment
    (root-mean-square) of their recent training losses.

    Timesteps with higher loss variance or magnitude are sampled more
    frequently, directing the model's training capacity toward the most
    challenging noise levels. A small uniform mixing probability ensures
    that no timestep is completely ignored.

    This sampler requires a warm-up period during which each timestep must
    be observed at least `history_per_term` times. Until warm-up is complete,
    uniform sampling is used as a fallback.

    Args:
        diffusion:          The GaussianDiffusion object (provides T).
        history_per_term:   Number of recent loss values to store per timestep
                            for computing the second moment.
        uniform_prob:       Minimum probability mass reserved for uniform sampling
                            to prevent any timestep from being starved.
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        # Circular buffer storing the most recent losses for each timestep.
        # Shape: [T, history_per_term]
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # Count of observed losses per timestep (capped at history_per_term).
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        Compute sampling weights as the RMS of recent losses per timestep.

        Before warm-up is complete (not all timesteps have been observed
        history_per_term times), returns uniform weights as a fallback.

        After warm-up:
            w[t] = (1 - uniform_prob) * sqrt(mean(loss_history[t]^2)) / Z
                   + uniform_prob / T
        where Z is a normalizing constant.
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        # Root-mean-square of stored losses for each timestep.
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        # Mix with uniform distribution to ensure minimum exploration.
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Record new loss observations into the circular history buffer.

        For each (timestep, loss) pair, the loss is appended to the buffer.
        Once the buffer is full, the oldest entry is shifted out (FIFO).

        Args:
            ts:     List of integer timesteps.
            losses: List of float losses corresponding to each timestep.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Buffer full: shift left and insert new loss at the end.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # Buffer not yet full: append at the current count position.
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check whether all timesteps have accumulated enough loss history
        to compute reliable second-moment estimates.
        """
        return (self._loss_counts == self.history_per_term).all()