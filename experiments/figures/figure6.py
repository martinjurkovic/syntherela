import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from syntherela.metadata import Metadata

sns.set(style="whitegrid")

metadata = Metadata(
).load_from_json("data/original/airbnb-simplified_subsampled/metadata.json")

methods = [
    'MOSTLYAI',
    'RGCLD',
    'CLAVADDPM',
    'RCTGAN',
    'REALTABFORMER',
    'SDV',
]

os.makedirs("results/figures/dcr", exist_ok=True)
table = "sessions"
bins = 10
for i, method in enumerate(methods):
    dcrs_real = torch.load(
        f"results/dcr/{table}_{method}_dcrs_real.pt", weights_only=True
    ).numpy()
    dcrs_test = torch.load(
        f"results/dcr/{table}_{method}_dcrs_test.pt", weights_only=True
    ).numpy()
    bins = np.histogram_bin_edges(
        np.concatenate((dcrs_real, dcrs_test)), bins=bins
    )

    plt.hist(dcrs_real, bins=bins, label='Train', density=True, color='blue')
    plt.hist(
        dcrs_test,
        bins=bins,
        alpha=.5,
        label='Holdout',
        density=True,
        color='orange'
    )

    plt.yscale('log')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('DCR', fontsize=20)
    plt.ylabel('Frequency(log)', fontsize=20)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig(
        f"results/figures/dcr/figure6{chr(ord('`')+(i+1))}.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.clf()
