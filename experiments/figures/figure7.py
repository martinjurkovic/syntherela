import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sdmetrics.single_table.detection import LogisticDetection

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns
from syntherela.metrics.single_table.detection.single_table_detection import (
    SingleTableDetection,
)

sns.set_theme()


def load_data(dataset_name, target_table):
    metadata = Metadata().load_from_json(f"data/original/{dataset_name}/metadata.json")
    tables = load_tables(f"data/original/{dataset_name}/", metadata)
    tables, metadata = remove_sdv_columns(tables, metadata)

    table_meta = metadata.get_table_meta(target_table, to_dict=True)

    return tables, table_meta, metadata


def symulate_generation(tables, target_table, seed=None):
    table = tables[target_table]
    table_perfect, table_original = train_test_split(
        table, test_size=0.5, random_state=seed
    )
    table_shuffled = table_perfect.copy()
    for column in table_shuffled.columns:
        table_shuffled[column] = table_shuffled[column].sample(frac=1).values
    return table_perfect, table_original, table_shuffled


def initialize_metrics(seed):
    xgb_cls = xgb.XGBClassifier
    xgb_args = {"seed": seed}
    lin_cls = LogisticRegression
    lin_args = {"random_state": seed}

    dd_xgb = SingleTableDetection(
        classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=seed
    )
    ld = SingleTableDetection(
        classifier_cls=lin_cls, classifier_args=lin_args, random_state=seed
    )
    return dd_xgb, ld


datasets = [
    "airbnb-simplified_subsampled",
    "Biodegradability_v1",
    "imdb_MovieLens_v1",
    "rossmann_subsampled",
    "walmart_subsampled",
]

dataset_names = {
    "airbnb-simplified_subsampled": "Airbnb\n(users)",
    "Biodegradability_v1": "Biodegradability\n(molecule)",
    "imdb_MovieLens_v1": "IMDB\n(movies)",
    "rossmann_subsampled": "Rossmann\n(store)",
    "walmart_subsampled": "Walmart\n(depts)",
}

target_tables = ["users", "molecule", "movies", "store", "stores"]

## Data Copying


def symulate_data_copying(tables, target_table, seed=None, frac_copied=1.0):
    table = tables[target_table]
    table_perfect, table_original = train_test_split(
        table, test_size=0.5, random_state=seed
    )
    table_copied = pd.concat(
        [
            table_original.sample(frac=frac_copied),
            table_perfect.sample(frac=1 - frac_copied),
        ]
    )
    return table_perfect, table_original, table_copied


seed = 0
results = {}
for dataset_name, target_table in zip(datasets, target_tables):
    tables, table_meta, _ = load_data(dataset_name, target_table)

    dd_xgb_perfect = []
    dd_xgb_copied = []
    ld_perfect = []
    ld_copied = []

    for i in tqdm(range(100), desc=dataset_name):
        # "generate" data
        table_perfect, table_original, table_copied = symulate_data_copying(
            tables, target_table, seed=seed + i
        )
        # prepare metrics
        dd_xgb, _ = initialize_metrics(seed + i)
        ld = LogisticDetection()

        # DD(XGB)
        results_dd_xgb_copied = dd_xgb.run(
            table_original,
            table_copied,
            table_meta,
        )

        # LD
        results_ld_copied = ld.compute(
            table_original,
            table_copied,
            table_meta,
        )

        dd_xgb_copied.append(results_dd_xgb_copied["accuracy"])
        ld_copied.append((results_ld_copied - 2) / -2)

    results[dataset_name] = {
        "dd_xgb_perfect": dd_xgb_perfect,
        "dd_xgb_copied": dd_xgb_copied,
        "ld_perfect": ld_perfect,
        "ld_copied": ld_copied,
    }

copying_behaviour = {}
for dataset_name, target_table in zip(datasets, target_tables):
    tables, table_meta, _ = load_data(dataset_name, target_table)

    accuracies = []
    std = []
    fractions = []
    step = -0.1
    for f in tqdm(np.arange(1, step, step), desc=dataset_name):
        f = round(max(0.0, f), 2)
        accs = []
        for i in range(100):
            _, table_original, table_copied = symulate_data_copying(
                tables, target_table, seed=seed + i, frac_copied=f
            )
            results_dd_xgb_copied = dd_xgb.run(
                table_original,
                table_copied,
                table_meta,
            )
            accs.append(results_dd_xgb_copied["accuracy"])
        accuracies.append(np.mean(accs))
        std.append(np.std(accs))
        fractions.append(f)
    copying_behaviour[dataset_name] = {
        "accuracies": np.array(accuracies),
        "std": np.array(std),
        "fractions": np.array(fractions),
    }

colormap = plt.cm.tab20
fig, axes = plt.subplots(1, 2, figsize=(10, 6))


def plot(ax, results, i, color, width=0.48, label=""):
    mean = np.mean(results)
    se = np.std(results) / np.sqrt(len(results))
    ax.bar(i, mean, label=label, width=width, color=color)
    ax.errorbar(i, mean, yerr=se, fmt="", color="black")


for i, dataset in enumerate(datasets):
    if i == 0:
        label1 = "DD(XGB)"
        label2 = "LD - max(AUC, 1/2)"
    else:
        label1 = ""
        label2 = ""

    plot(axes[0], results[dataset]["ld_copied"], 0 + i * 1.5, colormap(1), label=label2)
    plot(
        axes[0],
        results[dataset]["dd_xgb_copied"],
        0.5 + i * 1.5,
        colormap(3),
        label=label1,
    )


axes[0].set_xticks([0.25, 1.75, 3.25, 4.75, 6.25])
axes[0].set_xticklabels([dataset_names[dataset] for dataset in datasets], rotation=45)
axes[0].hlines(0.5, -0.25, 6.8, color="red", linestyle="--")
axes[0].legend(loc="upper right", bbox_to_anchor=(0.7, 1))
axes[0].set_ylim(0, 1)


axes[0].set_ylabel("Classification Accuracy")
axes[0].set_title("Copied Data")

for dataset_name in datasets:
    accuracies = copying_behaviour[dataset_name]["accuracies"]
    std = np.array(copying_behaviour[dataset_name]["std"])
    fractions = copying_behaviour[dataset_name]["fractions"]
    axes[1].errorbar(
        fractions,
        accuracies,
        yerr=std / np.sqrt(10),
        fmt="o",
        label=dataset_names[dataset_name].split("\n")[0],
    )

axes[1].hlines(0.5, 0, 1, color="red", linestyle="--")
axes[1].set_xticks(np.arange(0, 1.1, 0.1))
axes[1].set_title("Copying Behaviour by Dataset")
axes[1].set_xlabel("Fraction of Copied Data")
axes[1].set_ylabel("Classification Accuracy")
axes[1].legend()

fig.tight_layout()
plt.savefig("results/figures/figure7.png", dpi=300)
plt.show()
