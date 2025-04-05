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

from torch.utils.data import DataLoader
from DelayNet import DelayNetwork

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


def train(
    model: DelayNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    pbar: bool = False,
    with_bf_loss=True,
    max_norm=0.01,
    bf_reg_factor=0.001,
    extra_bf_loss: bool=False,
):
    model.train()
    loss_sum = 0.0

    progress = tqdm(dataloader, desc="Processing epoch", disable=not pbar)
    for X_batch, y_batch in progress:
        X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()

        mem_pot_out, rec_spike_raster = model(X_batch)

        losses = [
            F.cross_entropy(mem_pot_out, y_batch),
        ]
        in_spike_count = X_batch.sum()
        spike_count = rec_spike_raster.sum()

        if with_bf_loss:
            temporal_spike_count = torch.sum(rec_spike_raster, dim=(0, 1))
            approx_branch_factor_error = torch.diff(temporal_spike_count).pow(2).mean()
            losses.append(bf_reg_factor * approx_branch_factor_error)
            
            if extra_bf_loss:
                extra = (
                    bf_reg_factor
                    * torch.relu(1.2 * in_spike_count - spike_count)
                    / X_batch.shape[0]
                )
                losses.append(extra)

        loss = sum(losses)
        loss.backward()

        try:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm, error_if_nonfinite=True
            )
        except RuntimeError:
            print(
                "Gradient is non-finite - Downscaling parameters and processing next epoch"
            )
            model.downscale_parameters()
            model.zero_grad()
            optimizer.defaults
            for param_group in optimizer.param_groups:
                old_lr = float(param_group["lr"])
                new_lr = max(old_lr * 0.8, 1e-6)
                param_group["lr"] = new_lr
                for p in param_group["params"]:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        if "grad_sum_sq" in state:
                            state["grad_sum_sq"] = torch.zeros_like(
                                p
                            )  # Reset second moment
                        if "grad_sum" in state:
                            state["grad_sum"] = torch.zeros_like(
                                p
                            )  # Reset first moment
                        if "momentum_buffer" in state:
                            state["momentum_buffer"] = torch.zeros_like(
                                p
                            )  # Reset momentum buffer

            continue

        optimizer.step()
        model.clamp_parameters()  # Make sure params are bounded

        loss_batch = loss.detach().cpu().item() * len(y_batch)
        loss_sum += loss_batch

        progress.set_postfix(
            loss=loss_batch,
            spike_count=spike_count.detach().cpu().item(),
            in_spike_count=in_spike_count.cpu().item(),
        )

    return loss_sum


def eval(
    model: DelayNetwork,
    dataloader: DataLoader,
    device: torch.device,
    pbar: bool = False,
    w_spike_rasters=False,
):
    model.eval()
    with torch.no_grad():
        loss_sum = 0.0
        correct_preds = 0
        nb_elements = 0
        for X_batch, y_batch in tqdm(dataloader, disable=not pbar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            mem_pot_out, rec_spike_raster = model(X_batch)
            if w_spike_rasters:
                return rec_spike_raster.detach().cpu()

            loss = F.cross_entropy(mem_pot_out, y_batch)
            loss_sum += loss.detach().cpu().item() * len(y_batch)
            preds = torch.argmax(mem_pot_out, dim=1)
            correct_preds += (preds == y_batch).sum().detach().cpu().item()
            nb_elements += len(y_batch)

        return loss_sum, correct_preds / nb_elements
