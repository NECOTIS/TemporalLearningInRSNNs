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

import os
from uuid import uuid4

from CueAccumulationDataset import CueAccumulationDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from fire import Fire
from madgrad import MADGRAD
from tqdm import trange, tqdm


from DelayNet import DelayNetwork, SpikeFunc, get_delays


EXP_ID = uuid4().hex
EXP_NAME = "cue_accumulation_task"


def to_scratch(path):
    os.makedirs(f"logs/{EXP_NAME}/{EXP_ID}", exist_ok=True)
    return os.path.join(f"logs/{EXP_NAME}/{EXP_ID}", path)


def train(
    model: DelayNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    pbar: bool = False,
    with_bf_loss=True,
    max_norm=0.01,
    bf_reg_factor=0.001,
):

    model.train()
    loss_sum = 0.0

    pbar = tqdm(dataloader, desc="Processing epoch", disable=not pbar)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        X_batch = torch.swapaxes(X_batch, 1, 2)
        y_batch = y_batch[:, -1]
        optimizer.zero_grad()

        mem_pot_out, rec_spike_raster, _ = model(X_batch)
        spike_count = torch.sum(rec_spike_raster)

        temporal_spike_count = torch.sum(rec_spike_raster, dim=(0, 1))
        temporal_spike_count_rolled = torch.roll(temporal_spike_count, -1)
        approx_branch_factor_error = F.mse_loss(
            temporal_spike_count, temporal_spike_count_rolled
        )

        losses = [
            F.cross_entropy(mem_pot_out, y_batch),
        ]

        if with_bf_loss:
            losses.append(bf_reg_factor * approx_branch_factor_error)

        loss = sum(losses)
        loss.backward()

        try:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_norm, error_if_nonfinite=True
            )
        except RuntimeError:
            print("Gradient is non-finite")
            exit(1)

        optimizer.step()
        model.clamp_parameters()

        if spike_count <= 5:
            print("Spike count critically low")
            exit(1)

        loss_sum += loss.detach().cpu().item() * len(y_batch)

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
        if w_spike_rasters:
            spike_rasters = []
            mem_pot_out_batches = []
        for X_batch, y_batch in tqdm(dataloader, disable=not pbar):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            X_batch = torch.swapaxes(X_batch, 1, 2)
            y_batch = y_batch[:, -1]

            mem_pot_out, rec_spike_raster, mem_pot_out_history = model(X_batch)
            if w_spike_rasters:
                spike_rasters.append(rec_spike_raster.detach().cpu())
                mem_pot_out_batches.append(mem_pot_out_history)

            loss = F.cross_entropy(mem_pot_out, y_batch)
            loss_sum += loss.detach().cpu().item() * len(y_batch)
            preds = torch.argmax(mem_pot_out, dim=1)
            correct_preds += (preds == y_batch).sum().detach().cpu().item()
            nb_elements += len(y_batch)

        if w_spike_rasters:
            return (
                spike_rasters,
                loss_sum,
                correct_preds / nb_elements,
                mem_pot_out_batches,
            )
        return loss_sum, correct_preds / nb_elements


def main(
    seed=0x1B,
    n_rec=125,
    train_length=1024,
    test_length=500,
    nb_epochs=500,
    with_bf_loss=True,
    n_cues=7,
    lr=1e-3,
    max_norm=0.01,
    bf_reg_factor=0.001,
    initial_weights=None,
    init_sr=None,
    momentum=0.9,
    batch_size=64,
):

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    delays = get_delays(n_rec=125, ratio_1=0.3, ratio_2=0.2, delay_1=20, delay_2=80)

    val, counts = np.unique(delays, return_counts=True)
    print("Distribution of delays:", val, counts / np.sum(counts))

    delays = torch.as_tensor(delays).to(device)
    model = DelayNetwork(
        n_in=40,
        n_rec=n_rec,
        n_out=2,
        spike_func=SpikeFunc.apply,
        dt=0.001,
        delays=delays,
    )
    if initial_weights is not None:
        if init_sr > 0:
            model.scale_weights(init_sr)
        model.load_state_dict(torch.load(initial_weights))
    else:
        torch.save(model.state_dict(), to_scratch("initial_parameters.pt"))

    model = model.to(device)

    train_set = CueAccumulationDataset(train_length, n_cues=n_cues)
    loader_args = {
        "pin_memory": True if torch.cuda.is_available() else None,
        "batch_size": batch_size,
    }
    train_loader = DataLoader(train_set, **loader_args)
    val_set = CueAccumulationDataset(test_length, n_cues=n_cues)
    val_loader = DataLoader(val_set, **loader_args)

    optimizer = MADGRAD(model.parameters(), lr=lr, momentum=momentum)
    loss_history = []
    max_train_acc = 0
    pbar = trange(nb_epochs, desc="Training")
    for i in pbar:
        train_set.generate_samples()
        loss_history.append(
            train(
                model,
                train_loader,
                optimizer,
                device,
                True,
                with_bf_loss,
                max_norm,
                bf_reg_factor,
            )
        )
        _, acc = eval(model, train_loader, device)
        print(f"Epoch {i} -- Train acc={acc}; Train loss={loss_history[-1]/train_length}")
        if acc > max_train_acc:
            print("New maximum accuracy -- saving model")
            max_train_acc = acc
            torch.save(model.state_dict(), to_scratch("best_model.pt"))
            np.savetxt(to_scratch("best_train_acc.txt"), [acc])
            _, acc = eval(model, val_loader, device)
            print(f"valid acc={acc}")
            np.savetxt(to_scratch("valid_acc.txt"), [acc])

            if max_train_acc >= 1.0:
                break

    test_set = CueAccumulationDataset(test_length, n_cues=n_cues)
    test_loader = DataLoader(test_set, **loader_args)
    _, test_acc = eval(model, test_loader, device)
    print(f"final test acc = {test_acc}")
    np.savetxt(to_scratch("test_acc.txt"), [test_acc])
    np.savetxt(to_scratch("loss_history.txt"), loss_history)


def rnd_main():
    return main(
        lr=np.random.choice([1e-3, 1e-4, 1e-5]),
        max_norm=np.random.choice([1.0, 0.1, 0.01]),
        bf_reg_factor=np.random.choice([0.01, 0.001, 0.0001]),
        init_sr=np.random.uniform(1.0, 1.7),
        momentum=np.random.choice([0.9, 0.5, 0.0]),
        batch_size=int(np.random.choice([32, 64, 128])),
    )


def test(
    weights="weights/trained_125.pt", n_rec=125, n_cues=7, t_wait=1200, test_size=2048
):
    seed = 0x1B
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    delays, mesh = get_delays(return_pos=True)
    delays = torch.as_tensor(delays).to(device)

    model = DelayNetwork(
        n_in=40,
        n_rec=n_rec,
        n_out=2,
        spike_func=SpikeFunc.apply,
        dt=0.001,
        delays=delays,
    )

    model.load_state_dict(torch.load(weights))
    model = model.eval().to(device)

    tau_rec = model.tau_rec.detach().cpu().numpy()[0]
    tau_out = model.tau_out.detach().cpu().numpy()[0]
    thr_rec = model.thr_rec.detach().cpu().numpy()[0]

    print("Tau rec", tau_rec / (0.001 * 10))
    print("Tau out", tau_out / (0.001 * 10))
    print("thr_rec", thr_rec)

    W_r = model.W_r.detach().cpu().numpy()
    print("Spectral radius:", np.max(np.abs(np.linalg.eigvals(W_r))))
    print("Mean W_r", W_r.mean())
    print("Max W_r", W_r.max())
    W_i = model.W_i.detach().cpu().numpy()
    print("Mean W_i", W_i.mean())
    print("Max W_i", W_i.max())

    test_set = CueAccumulationDataset(test_size, n_cues=n_cues, t_wait=t_wait)
    loader_args = {
        "pin_memory": True if torch.cuda.is_available() else None,
        "batch_size": 32,
    }
    test_loader = DataLoader(test_set, **loader_args)
    _, acc = eval(model, test_loader, device)

    print(f"Test Accuracy={acc}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) <= 1:
        main()
    else:
        Fire()
