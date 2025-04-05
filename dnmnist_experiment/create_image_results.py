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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sns.set(style="whitegrid", font_scale=1.5)
    df = pd.read_csv("weights/experiments.csv")

    def get_label(row):
        if row.with_delays and row.with_bf_loss:
            return "SD+BFR"
        elif row.with_delays:
            return "Only SD"
        elif row.with_bf_loss:
            return "Only BFR"
        else:
            return "No BFR, No SD"

    df["label"] = df.apply(get_label, axis=1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax = sns.lineplot(
        df,
        x="sample_duration",
        y="test_acc",
        hue="label",
        markers=True,
        style="label",
        ax=ax,
        alpha=0.8,
    )
    ax.set(xlabel="Sample duration [ms]", ylabel="Test accuracy")
    ax.set_xticks([300, 350, 400, 450, 500])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), handles=handles, labels=labels)

    fig = ax.get_figure()
    plt.tight_layout()
    fig.savefig("dnmnist_results.pdf", bbox_inches="tight")

    plt.show()
