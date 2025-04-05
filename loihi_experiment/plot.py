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
# Date: March 26th, 2025
# Organization: Groupe de recherche en Neurosciences Computationnelles et Traitement Intelligent des Signaux (NECOTIS),
# Université de Sherbrooke, Canada

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr

sns.set(style="ticks", font_scale=1.5)
with open("res.json") as f:
    data = json.load(f)

    values = []
    for freq, freq_data in data.items():
        for delay, delay_data in freq_data.items():
            for metric, metric_data in delay_data.items():
                metric_data["delay"] = delay
                metric_data["freq"] = freq
                metric_data["metric"] = metric
                values.append(metric_data)

    df = pd.DataFrame(values)
    df["delay"] = df["delay"].astype(int)

    metrics = [
        "dynamic_energy",
        "static_energy",
        "total_time",
    ]
    plot_metadata = {
        "dynamic_energy": {"title": "(A) Dynamic Energy", "yaxis": "µJ"},
        "static_energy": {"title": "(B) Static Energy", "yaxis": "µJ"},
        "total_time": {"title": "(C) Total Time", "yaxis": "µS"},
    }
    metric_dfs = df.groupby("metric")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)
    for ax, metric in zip(axs, metrics):
        metric_df = metric_dfs.get_group(metric)
        metadata = plot_metadata[metric]
        ax.set_title(metadata["title"])
        ax.set_xlabel("Synaptic delay [ms]")
        ax.set_ylabel(metadata["yaxis"])

        for i, freq in enumerate(
            ["25", "50", "100"]
        ):  # order is important for joined legend
            freq_df = metric_df[freq == metric_df.freq]
            ax.errorbar(
                freq_df["delay"],
                freq_df["mean"] / 1e7 * 1e6,
                yerr=freq_df["std"] / 1e7 * 1e6,
                capsize=2,
                label=f"{freq} Hz",
                fmt=".-",
            )
            r = pearsonr(freq_df["delay"], freq_df["mean"])
            tau = kendalltau(freq_df["delay"], freq_df["mean"])
            print(f"Metric={metric}; Freq={freq}, tau={tau}, r={r}")

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))

    fig.savefig("loihi_results.png", bbox_inches="tight")
    fig.savefig("loihi_results.pdf", bbox_inches="tight")
