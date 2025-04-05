# Cue Accumulation Dataset
# Adapted from https://github.com/ChFrenkel/eprop-PyTorch
# Which was adapted from https://github.com/IGITUGraz/eligibility_propagation
# See License at https://github.com/ChFrenkel/eprop-PyTorch/blob/main/LICENSE

# Previously published in [C. Frenkel and G. Indiveri, "ReckOn: A 28nm sub-mm² task-agnostic spiking recurrent neural network processor enabling on-chip learning over second-long timescales," IEEE International Solid-State Circuits Conference (ISSCC), 2022]
# And [G. Bellec et al., "A solution to the learning dilemma for recurrent networks of spiking neurons," Nature communications, vol. 11, no. 3625, 2020]

import torch
from torch.utils import data
import numpy as np
import numpy.random as rd


class CueAccumulationDataset(data.Dataset):

    def __init__(self, length, n_cues=7, t_wait=1200):

        self.length = length
        self.n_cues = n_cues
        self.f0 = 40
        self.t_cue = 100
        self.t_wait = t_wait

        self.n_symbols = 4  # 2 bits + noise + recall cue
        self.p_group = 0.3
        self.dt = 1e-3
        self.t_interval = 150
        self.seq_len = self.n_cues * self.t_interval + self.t_wait
        self.n_in = 40  # Total nb of neurons
        self.n_out = 2

        self.generate_samples()

    def generate_samples(self):
        n_channel = self.n_in // self.n_symbols
        prob0 = self.f0 * self.dt
        t_silent = self.t_interval - self.t_cue

        # Randomly assign group A and B
        prob_choices = np.array([self.p_group, 1 - self.p_group], dtype=np.float32)
        idx = rd.choice([0, 1], self.length)
        probs = np.zeros((self.length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]

        cue_assignments = np.zeros((self.length, self.n_cues), dtype=int)
        # For each example in batch, draw which cues are going to be active (left or right)
        for b in range(self.length):
            cue_assignments[b, :] = rd.choice([0, 1], self.n_cues, p=probs[b])

        # Generate input spikes
        input_spike_prob = np.zeros((self.length, self.seq_len, self.n_in))
        t_silent = self.t_interval - self.t_cue
        for b in range(self.length):
            for k in range(self.n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[
                    b,
                    t_silent
                    + k * self.t_interval : t_silent
                    + k * self.t_interval
                    + self.t_cue,
                    c * n_channel : (c + 1) * n_channel,
                ] = prob0

        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval :, 2 * n_channel : 3 * n_channel] = prob0
        input_spike_prob[:, :, 3 * n_channel :] = prob0 / 4.0
        input_spikes = generate_poisson_noise_np(input_spike_prob)
        self.x = torch.tensor(input_spikes).float()

        # Generate targets
        target_nums = np.zeros((self.length, self.seq_len), dtype=int)
        target_nums[:, :] = np.transpose(
            np.tile(
                np.sum(cue_assignments, axis=1) > int(self.n_cues / 2),
                (self.seq_len, 1),
            )
        )
        self.y = torch.tensor(target_nums).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [
            generate_poisson_noise_np(pb, freezing_seed=freezing_seed)
            for pb in prob_pattern
        ]

    shp = prob_pattern.shape

    if freezing_seed is not None:
        rng = rd.RandomState(freezing_seed)
    else:
        rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes
