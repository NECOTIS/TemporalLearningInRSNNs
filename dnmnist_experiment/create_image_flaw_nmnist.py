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

import matplotlib.pyplot as plt
import os
from ebdataset.vision import NMnist
from ebdataset.vision.transforms import ToDense
from ebdataset import ms
import numpy as np

if __name__ == "__main__":
    dt = 1 * ms
    dataset_path = os.environ.get("NMNIST_DATASET_PATH", "./data/nmnist")
    samples = NMnist(dataset_path, is_train=True, transforms=ToDense(dt))

    fig, axs = plt.subplots(4, 4, tight_layout=True)
    axs = axs.flatten()
    np.random.seed(0x1B)
    samples_idx = np.random.choice(len(samples), size=16)
    memory_requirement = 30
    for i, idx in enumerate(samples_idx):
        spike_train, label = samples[idx]
        # spike_train of shape [34, 34, 2, ~300]
        spike_count = np.sum(
            spike_train.numpy()[:, :, :, -memory_requirement:], axis=-1
        )

        rgb = np.zeros((34, 34, 3))
        rgb[:, :, 1:] = np.clip(spike_count, 0, 1).astype(float)

        axs[i].imshow(np.swapaxes(rgb, 0, 1))
        axs[i].set_title(f"Label={label}")
        axs[i].grid(False)
        axs[i].axis("off")

    fig.savefig(f"nmnist_digits_last_{memory_requirement}ms.pdf", bbox_inches="tight")
    plt.show()
