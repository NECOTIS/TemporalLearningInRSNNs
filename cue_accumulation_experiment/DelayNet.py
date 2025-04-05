# Copyright (c) 2025, NECOTIS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Ismael Balafrej
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import trange


def get_delays(
    n_rec=125, ratio_1=0.3, ratio_2=0.2, delay_1=20, delay_2=80, return_pos=False
):
    from scipy.spatial import distance_matrix

    x = int(np.round(n_rec ** (1 / 3)))
    assert x**3 == n_rec

    width, height, depth = x, x, x
    mesh = np.mgrid[:width, :height, :depth].astype(float) + np.random.uniform(
        0.0, 0.5, size=(3, width, height, depth)
    )
    x, y, z = mesh
    mesh_flat = mesh.reshape(3, -1).T
    center = np.asarray(((width / 2, height / 2, depth / 2),))
    distance_to_center = distance_matrix(center, mesh_flat)[0]
    mesh_flat -= center
    max_dist = np.sqrt(width**2 + height**2 + depth**2)
    delay_matrix = np.zeros((len(mesh_flat), len(mesh_flat)))
    delay_matrix[:, distance_to_center >= ratio_1 * max_dist] += delay_1
    delay_matrix[:, distance_to_center >= ratio_2 * max_dist] += delay_2

    # All the rest but the diagonal get a delay of 1 timestep
    delay_matrix[delay_matrix == 0] = 1
    np.fill_diagonal(delay_matrix, 0) # no self connection

    delay_matrix = np.ceil(delay_matrix).astype(int)

    if return_pos:
        return delay_matrix, mesh_flat
    else:
        return delay_matrix


class SpikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        ctx.save_for_backward(v_scaled)
        z_ = torch.gt(v_scaled, 0.0).float()
        return z_

    @staticmethod
    def backward(ctx, grad_output):
        (v_scaled,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[v_scaled < 0] *= F.relu(1 + v_scaled[v_scaled < 0])
        return grad_input


class DelayNetwork(nn.Module):
    def __init__(
        self,
        n_in,
        n_rec,
        n_out,
        delays,
        spike_func=SpikeFunc.apply,
        dt=0.001,
        with_heterogenous_time_constants=False,
        with_heterogenous_thresholds=False,
    ):
        super(DelayNetwork, self).__init__()
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.delays = delays.int()
        self.possible_delays = torch.unique(self.delays)
        self.possible_delays = self.possible_delays[
            self.possible_delays != 0
        ].contiguous()
        self.delay_masks = [self.delays == d for d in self.possible_delays]

        self.spike_func = spike_func
        self.dt = dt

        self.W_i = nn.Parameter(torch.empty(self.n_in, self.n_rec))
        self.W_r = nn.Parameter(torch.empty(self.n_rec, self.n_rec))
        self.W_o = nn.Parameter(torch.empty(self.n_rec, self.n_out))
        self.B_r = nn.Parameter(torch.empty(self.n_rec))
        self.B_o = nn.Parameter(torch.empty(self.n_out))

        self.with_heterogenous_time_constants = with_heterogenous_time_constants
        if with_heterogenous_time_constants:
            self.tau_rec = nn.Parameter(torch.empty(self.n_rec))
            self.tau_out = nn.Parameter(torch.empty(self.n_out))
        else:
            self.tau_rec = nn.Parameter(
                torch.empty(1)
            )  # Share time constant between rec neurons
            self.tau_out = nn.Parameter(
                torch.empty(1)
            )  # Share time constant between out neurons

        self.with_heterogenous_thresholds = with_heterogenous_thresholds
        if with_heterogenous_thresholds:
            self.thr_rec = nn.Parameter(torch.empty(self.n_rec))
        else:
            self.thr_rec = nn.Parameter(
                torch.empty(1)
            )  # Share LIF threshold between rec neurons

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_i.data, mode="fan_out")
        torch.nn.init.kaiming_uniform_(self.W_r.data)
        torch.nn.init.kaiming_uniform_(self.W_o.data)
        torch.zero_(self.B_r.data)
        torch.zero_(self.B_o.data)
        self.tau_rec.data.copy_(0.45 * torch.ones_like(self.tau_rec.data))
        self.tau_out.data.copy_(0.45 * torch.ones_like(self.tau_out.data))
        self.thr_rec.data.copy_(0.5 * torch.ones_like(self.thr_rec.data))

    def load_state_dict(self, state_dict, strict: bool = True):
        if "delays" in state_dict:
            self.delays = state_dict["delays"]
            self.possible_delays = torch.unique(self.delays)
            self.possible_delays = self.possible_delays[
                self.possible_delays != 0
            ].contiguous()
            self.delay_masks = [self.delays == d for d in self.possible_delays]

        return super().load_state_dict(state_dict, strict)

    def clamp_parameters(self):
        self.tau_rec.data.clamp_(
            1e-3, 0.5
        )  # Clamp time constant between 0.1 ms to 50ms
        self.tau_out.data.clamp_(
            1e-3, 0.5
        )  # Clamp time constant between 0.1 ms to 50ms
        self.thr_rec.data.clamp_(0.1, 100.0)  # Clamp threshold

        self.W_r.data *= 1 - torch.eye(
            self.n_rec, device=self.W_r.device
        )  # Make sure there is no self connection

        self.W_i.data = torch.abs(self.W_i.data)

    def downscale_parameters(self):
        self.W_r.data *= 0.99
        self.B_r.data *= 0.99

    def scale_weights(self, target_spectral_radius):
        factor = torch.max(torch.abs(torch.linalg.eigvals(torch.abs(self.W_r.data))))
        self.W_r.data *= target_spectral_radius / factor

    def forward(self, input_spike_raster, pbar=False):
        """Forward function for an input spike tensor"""
        device = input_spike_raster.device
        batch_size, n_in, sample_duration = input_spike_raster.shape
        mem_pot_rec = torch.zeros(
            batch_size, self.n_rec, device=device
        )  # Leaky integrate and fire neurons
        mem_pot_out = torch.zeros(
            batch_size, self.n_out, device=device
        )  # Leaky integrators
        rec_spike_raster = [None for _ in range(sample_duration)]

        # Multiply dt by 10 to have ms level tau at first decimal point
        cst = torch.tensor(-self.dt * 10, device=device)
        self.rec_time_const = torch.exp(cst / self.tau_rec)
        self.out_time_const = torch.exp(cst / self.tau_out)
        mem_pot_out_history = []

        for t in trange(sample_duration, disable=not pbar):
            # Integrate input currents
            input_spikes = input_spike_raster[:, :, t]

            mem_pot_rec, mem_pot_out, rec_spike_raster = self.forward_timestep(
                t, input_spikes, rec_spike_raster, mem_pot_rec, mem_pot_out
            )
            mem_pot_out_history.append(mem_pot_out.detach().cpu())

        return (
            mem_pot_out,
            torch.stack(rec_spike_raster, dim=2),
            torch.stack(mem_pot_out_history, dim=1),
        )

    def forward_timestep(
        self, t, input_spikes, rec_spike_raster, mem_pot_rec, mem_pot_out
    ):
        """Forward function for a single timestep"""
        input_current = input_spikes @ self.W_i + self.B_r
        input_current = input_current.detach()

        # Delay dynamics
        for i, d in enumerate(self.possible_delays):
            if t > d:  # 0 delay is diagonal
                delayed_spikes = rec_spike_raster[t - d]
                input_current = input_current + delayed_spikes @ (
                    self.W_r * self.delay_masks[i]
                )

        # LIF Dynamics
        mem_pot_rec = mem_pot_rec * self.rec_time_const + input_current

        # Apply spike function
        rec_spike_raster[t] = self.spike_func(
            mem_pot_rec - self.thr_rec
        )

        # Soft reset
        mem_pot_rec = mem_pot_rec - rec_spike_raster[t] * self.thr_rec

        # Integrate for output
        output_current = rec_spike_raster[t] @ self.W_o + self.B_o
        mem_pot_out = mem_pot_out * self.out_time_const + output_current

        return mem_pot_rec, mem_pot_out, rec_spike_raster
