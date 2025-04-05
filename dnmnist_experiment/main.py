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
from common import eval, train
from fire import Fire
from madgrad import MADGRAD
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from ebdataset.vision import NMnist
from ebdataset.vision.transforms import ToDense
from ebdataset import ms
import torch.nn.functional as F
import pandas as pd

from DelayNet import DelayNetwork, get_delays

EXP_ID = uuid4().hex
EXP_NAME = "delayed-n-mnist"
N_IN = 34 * 34
N_OUT = 10


def to_scratch(path):
    os.makedirs(f"logs/{EXP_NAME}", exist_ok=True)
    return os.path.join(f"logs/{EXP_NAME}", path)


def main(
    initial_weights=None,
    nb_epochs=1,  # 50,
    batch_size=64,
    lr=0.001,
    n_rec=216,
    delay_1=25,
    delay_2=61,
    init_sr=None,
    max_norm=0.1,
    bf_reg_factor=0.0001,
    two_layer=False,
    with_heterogenous_thresholds=True,
    with_heterogenous_time_constants=False,
    momentum=0.9,
    test_only=False,
    sample_duration=600,
    bf_reg_factor_decay=False,
    noise_freq_hz=10,
):
    params = locals()
    print(params)

    if not test_only:
        with open(to_scratch(f"{EXP_ID}.txt"), "w") as f:
            json.dump(params, f, default=repr)

    print("Starting experiment with id", EXP_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if device.type == "cpu":
        from multiprocessing import cpu_count

        n_cores = min(cpu_count(), 8)
        print(f"Using CPU {n_cores} cores")
        torch.set_num_threads(n_cores)

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
    if init_sr is not None:
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

    if not test_only:
        torch.save(model.state_dict(), to_scratch(f"initial_model_{EXP_ID}.pt"))

    train_loader, valid_loader, test_loader = get_data(
        batch_size=batch_size,
        sample_duration=sample_duration,
        noise_freq_hz=noise_freq_hz,
    )

    writer = None
    if not test_only:
        optimizer = MADGRAD(model.parameters(), lr=lr, momentum=momentum)
        loss_history = []
        max_val_acc = 0
        val_accs = []

        writer = SummaryWriter(to_scratch(f"runs/{EXP_ID}"))
        # writer.add_hparams(params, {})

        current_bf_reg = bf_reg_factor

        for epoch in trange(nb_epochs, desc="Training"):
            with_bf_reg = bf_reg_factor > 0
            if bf_reg_factor_decay:
                current_bf_reg = bf_reg_factor * pow(10, -epoch / 10)

            loss_epoch = train(
                model,
                train_loader,
                optimizer,
                device,
                True,
                with_bf_reg,
                max_norm,
                current_bf_reg,
            )
            writer.add_scalar("Loss/train", loss_epoch, epoch)
            loss_history.append(loss_epoch)
            _, acc = eval(model, valid_loader, device)
            writer.add_scalar("Accuracy/valid", acc, epoch)
            val_accs.append(acc)
            with open(to_scratch(f"val_acc_{EXP_ID}.txt"), "a") as f:
                f.write(f"{acc}\n")
            if acc > max_val_acc:
                max_val_acc = acc
                torch.save(model.state_dict(), to_scratch(f"best_model_{EXP_ID}.pt"))
                print(f"New maximum accuracy -- saving model {acc}")
                _, test_acc = eval(model, test_loader, device, pbar=False)
                with open(to_scratch(f"test_acc_{EXP_ID}.txt"), "w") as f:
                    f.write(f"{test_acc}\n")

            elif (np.diff(val_accs)[-5:] == 0).all():  # 5 epochs with constant val acc
                print("Early stopping - accuracy is not increasing")
                break
            elif acc < max_val_acc * 0.5:
                print("Accuracy dropped by 50% -- stopping")
                break

    _, test_acc = eval(model, test_loader, device, pbar=False)

    print(f"Final test accuracy: {test_acc:.4f}")
    if writer is not None:
        writer.add_scalar("Accuracy/test", test_acc)
        writer.close()


def rnd_main(normal_rnn: bool = False):
    n_rec = 6**3
    if normal_rnn:
        main(
            batch_size=32,  # 96,
            lr=np.random.choice([0.001, 0.0001]),
            n_rec=n_rec,
            delay_1=0,
            delay_2=0,
            init_sr=np.random.choice([None, np.random.uniform(1.1, 1.4)]),
            max_norm=0.1,
            bf_reg_factor=0.0,
            with_heterogenous_thresholds=True,
            with_heterogenous_time_constants=False,
            momentum=0.9,
            test_only=False,
            sample_duration=600,
            noise_freq_hz=10,
        )
    else:
        main(
            batch_size=32,  # 96,
            lr=np.random.choice([0.001, 0.0001]),
            n_rec=n_rec,
            delay_1=np.random.randint(15, 30),
            delay_2=np.random.randint(60, 90),
            init_sr=np.random.choice([None, np.random.uniform(1.1, 1.4)]),
            max_norm=np.random.choice([0.5, 0.1, 0.01]),
            bf_reg_factor=np.random.choice([0.001, 0.0001]),
            with_heterogenous_thresholds=True,
            with_heterogenous_time_constants=False,
            momentum=0.9,
            test_only=False,
            sample_duration=600,
            bf_reg_factor_decay=False,
            noise_freq_hz=10,
        )


def test(exp_id: str):
    df = pd.read_csv("weights/experiments.csv")
    selection = df[df.exp_id == exp_id]
    if len(selection) == 0:
        print(
            "Unknown experiment id. Please provide one of the id in the weights directory."
        )
        return 1

    args = selection.to_dict("records")[0]
    return main(
        initial_weights=f"weights/{exp_id}.pt",
        test_only=True,
        n_rec=args["n_rec"],
        delay_1=args["delay_1"],
        delay_2=args["delay_2"],
        two_layer=args["two_layer"],
        with_heterogenous_thresholds=args["with_heterogenous_thresholds"],
        with_heterogenous_time_constants=args["with_heterogenous_time_constants"],
        sample_duration=args["sample_duration"],
        noise_freq_hz=args["noise_freq_hz"],
    )


class Collate:

    def __init__(self, noise_freq_hz, sample_duration):
        self.noise_freq_hz = noise_freq_hz
        self.sample_duration = sample_duration

    def __call__(self, samples):
        batch = []
        labels = []

        max_duration = max(s[0].shape[-1] for s in samples)
        batch_duration = None
        if self.sample_duration is None:
            with_noise = False
            batch_duration = max_duration
        else:
            with_noise = True
            batch_duration = self.sample_duration

        for tensor, label in samples:
            labels.append(label)
            # Only 1 polarity + flatten
            tensor = tensor[:, :, 0, :].view(-1, tensor.shape[-1])

            if tensor.shape[-1] < batch_duration:
                pad_duration = batch_duration - tensor.shape[-1]
                tensor = F.pad(tensor, (0, pad_duration), "constant", 0)
                if with_noise:
                    noise = np.random.poisson(
                        lam=self.noise_freq_hz / 1000,
                        size=(tensor.shape[0], pad_duration),
                    )
                    tensor[:, -pad_duration:] += torch.from_numpy(noise)
            if tensor.shape[-1] > batch_duration:
                tensor = tensor[:, :batch_duration]

            batch.append(tensor)

        return torch.stack(batch), torch.from_numpy(np.array(labels)).long()


def get_data(batch_size, sample_duration, noise_freq_hz):
    dataset_path = os.environ.get("NMNIST_DATASET_PATH", "./data/nmnist")

    # shape is (34, 34, 2, duration of sample)
    # Model expects shape (batch_size, n_in, sample_duration)

    dt = 1 * ms
    train = NMnist(dataset_path, is_train=True, transforms=ToDense(dt))
    train, valid = torch.utils.data.random_split(train, [0.9, 0.1])

    dataloader_kargs = {}
    if torch.cuda.is_available():
        dataloader_kargs["num_workers"] = 8
        dataloader_kargs["pin_memory"] = True
        dataloader_kargs["pin_memory_device"] = "cuda"

    collate_fn = Collate(noise_freq_hz, sample_duration)

    train_loader = torch.utils.data.DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dataloader_kargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dataloader_kargs,
    )

    test = NMnist(dataset_path, is_train=False, transforms=ToDense(dt))
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dataloader_kargs,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        main()
    else:
        Fire()
