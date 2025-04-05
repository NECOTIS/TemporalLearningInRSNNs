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

import json
import os
import sys
from uuid import uuid4

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from common import eval, train
from DelayNet import DelayNetwork, get_delays
from fire import Fire
from madgrad import MADGRAD
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns

EXP_ID = uuid4().hex
EXP_NAME = "sequential_permuted_mnist"
N_IN = 1
N_OUT = 10


def to_scratch(path):
    os.makedirs(f"logs/{EXP_NAME}", exist_ok=True)
    return os.path.join(f"logs/{EXP_NAME}", path)


def main(
    initial_weights=None,
    permutation=None,
    nb_epochs=100,
    batch_size=64,
    lr=0.001,
    n_rec=216,
    delay_1=25,
    delay_2=61,
    init_sr=1.25,
    max_norm=0.1,
    bf_reg_factor=0.0001,
    two_layer=False,
    with_heterogenous_thresholds=True,
    with_heterogenous_time_constants=False,
    momentum=0.9,
):
    params = locals()
    print(params)
    with open(to_scratch(f"{EXP_ID}.txt"), "w") as f:
        json.dump(params, f, default=repr)

    print("Starting experiment with id", EXP_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.backends.cudnn.benchmark = True

    delays = get_delays(
        n_rec=n_rec,
        ratio_1=0.3,
        ratio_2=0.2,
        delay_1=delay_1,
        delay_2=delay_2,
        return_pos=False,
    )
    delays = torch.as_tensor(delays).to(device)
    model = DelayNetwork(
        n_in=N_IN,
        n_rec=n_rec,
        n_out=N_OUT,
        delays=delays,
        two_layer=two_layer,
        with_heterogenous_thresholds=with_heterogenous_thresholds,
        with_heterogenous_time_constants=with_heterogenous_time_constants,
    ).to(device)
    if init_sr > 0:
        model.scale_weights(init_sr)

    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    print("Num of params: ", get_n_params(model))

    if initial_weights is not None:
        model.load_state_dict(torch.load(initial_weights))

    torch.save(model.state_dict(), to_scratch(f"initial_model_{EXP_ID}.pt"))

    perm = None if permutation is None else torch.load(permutation)
    train_loader, valid_loader, test_loader, perm = get_data(
        batch_size=batch_size, perm=perm
    )

    torch.save(perm, to_scratch(f"permutation_{EXP_ID}.pt"))

    optimizer = MADGRAD(model.parameters(), lr=lr, momentum=momentum)
    loss_history = []
    max_val_acc = 0
    val_accs = []

    writer = SummaryWriter(to_scratch(f"runs/{EXP_ID}"))
    # writer.add_hparams(params, {})

    for i in trange(nb_epochs, desc="Training"):
        loss_epoch = train(
            model, train_loader, optimizer, device, True, True, max_norm, bf_reg_factor
        )
        writer.add_scalar("Loss/train", loss_epoch, i)
        loss_history.append(loss_epoch)
        _, acc = eval(model, valid_loader, device)
        writer.add_scalar("Accuracy/valid", acc, i)
        val_accs.append(acc)
        with open(to_scratch(f"val_acc_{EXP_ID}.txt"), "a") as f:
            f.write(f"{acc}\n")
        if acc > max_val_acc:
            max_val_acc = acc
            torch.save(model.state_dict(), to_scratch(f"best_model_{EXP_ID}.pt"))
            print(f"New maximum accuracy -- saving model {acc}")
        elif (np.diff(val_accs)[-5:] == 0).all():  # 5 epochs with constant val acc
            print("Early stopping - accuracy is not increasing")
            break
        elif acc < max_val_acc * 0.5:
            print("Accuracy dropped by 50% -- stopping")
            break

    model.load_state_dict(torch.load(to_scratch(f"best_model_{EXP_ID}.pt")))
    _, acc = eval(model, test_loader, device)
    print(f"Test accuracy: {acc}")
    with open(to_scratch(f"test_acc_{EXP_ID}.txt"), "a") as f:
        f.write(f"{acc}\n")
    writer.add_scalar("Accuracy/test", acc)

    writer.close()


def rnd_main():
    main(
        initial_weights="models/mnist_trained_216.pt",
        permutation="models/mnist_permutation.pt",
        batch_size=64,
        lr=np.random.choice([0.001, 0.0001]),
        n_rec=216,
        delay_1=np.random.randint(15, 30),
        delay_2=np.random.randint(60, 90),
        init_sr=np.random.uniform(1.0, 1.7),
        max_norm=0.1,
        bf_reg_factor=np.random.choice([0.0001, 0.00001]),
        two_layer=False,
        with_heterogenous_thresholds=np.random.choice([True, False]),
        momentum=np.random.choice([0.9, 0.5, 0.0]),
    )


def test(
    weights: str = "weights/trained_216.pt",
    permutation: str = "weights/permutation.pt",
    batch_size=64,
    n_rec=216,
    delay_1=25,
    delay_2=61,
    two_layer=False,
    with_heterogenous_thresholds=True,
    with_heterogenous_time_constants=False,
):
    perm = torch.load(permutation)
    _, _, test_loader, _ = get_data(batch_size=batch_size, perm=perm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delays = get_delays(
        n_rec=n_rec,
        ratio_1=0.3,
        ratio_2=0.2,
        delay_1=delay_1,
        delay_2=delay_2,
        return_pos=False,
    )
    delays = torch.as_tensor(delays).to(device)
    model = DelayNetwork(
        n_in=N_IN,
        n_rec=n_rec,
        n_out=N_OUT,
        delays=delays,
        two_layer=two_layer,
        with_heterogenous_thresholds=with_heterogenous_thresholds,
        with_heterogenous_time_constants=with_heterogenous_time_constants,
    ).to(device)
    print(f"Network size: {sum(p.numel() for p in model.parameters())}")

    model.load_state_dict(torch.load(weights))
    print(model.tau_out, model.tau_rec)
    _, acc = eval(model, test_loader, device, True, False)
    print(f"Test accuracy: {acc}")
    spike_rasters = eval(model, test_loader, device, True, True)

    np.random.seed(0)
    sample_id = np.random.randint(0, len(spike_rasters))
    sample = spike_rasters[sample_id].numpy()

    sns.set(style="whitegrid", font_scale=2)

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    plt.eventplot([np.flatnonzero(i) for i in sample])
    plt.xlabel("Time [ms]")
    plt.ylabel("Neuron ID")
    fig.savefig("spike_raster_psmnist.pdf", bbox_inches="tight")

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    spike_counts = np.sum(sample, axis=0)

    plt.scatter(spike_counts[:-1], spike_counts[1:], s=2, alpha=1)
    plt.xlabel("$\\sum_i \\vec{n}_i[t]$")
    plt.ylabel("$\\sum_i \\vec{n}_i[t+1]$")
    x = np.linspace(0, spike_counts.max(), 1000)
    plt.plot(x, x, label="Branching factor of 1", color="black")
    fig.savefig("poincarre_psmnist.pdf", bbox_inches="tight")

    fig = plt.figure(figsize=(8, 8), tight_layout=True)
    for neuron_id in range(len(sample)):
        plt.plot(np.cumsum(sample[neuron_id]), alpha=0.7)
    plt.xlabel("Time [ms]")
    plt.ylabel("Cumulative spike count per neuron")
    fig.savefig("cumulative_spike_count_psmnist.pdf", bbox_inches="tight")

    spike_rate = np.mean(sample)
    print(f"Spike rate: {spike_rate}")

    plt.show()


def get_data(batch_size, perm=None):
    if perm is None:
        torch.manual_seed(0)  # Make sur randperm is always the same
        perm = torch.randperm(784)

    seq_perm_trans = transforms.Lambda(lambda x: x.view(-1)[perm].unsqueeze(0))

    train_transform = test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            seq_perm_trans,
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="data/", train=True, transform=train_transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="data/", train=False, transform=test_transform, download=True
    )

    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [55000, 5000]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, valid_loader, test_loader, perm


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        main()
    else:
        Fire()
