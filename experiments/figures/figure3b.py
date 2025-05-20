import numpy as np
import os
import xgboost as xgb
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection import AggregationDetection
from syntherela.data import load_tables

sns.set_theme(style="white")
rc("font", **{"family": "serif", "serif": ["Times"], "size": 30})
rc("text", usetex=True)

dataset_name = "Berka_subsampled"
method = "CLAVADDPM"

metadata = Metadata().load_from_json(f"data/original/{dataset_name}/metadata.json")

tables = load_tables(f"data/original/{dataset_name}/", metadata)
tables_synthetic = load_tables(
    f"data/synthetic/{dataset_name}/{method}/1/sample1", metadata
)

# Compute the metric
xgb_cls = xgb.XGBClassifier
xgb_args = {
    "seed": 0,
    "importance_type": "gain",
}

metric = AggregationDetection(
    classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
)

for table in tables.keys():
    tables_synthetic[table] = tables_synthetic[table][tables[table].columns]

metric.run(tables, tables_synthetic, metadata=metadata, target_table="account")

feature_importance = metric.feature_importance(combine_categorical=False)

# pair = ['features_Store_MarkDown3_mean', 'depts_Store_Dept_nunique'] # Walamrt
pair = ["trans_account_id_bank_nunique", "trans_account_id_counts"]
# feature_importance

# set colormap to coolwarm
cmap = sns.color_palette("coolwarm", as_cmap=True)


def prettyify_feature_name(feature_name):
    feature_name = feature_name.replace("trans", "transaction -")
    feature_name = feature_name.replace("TRANS", "TRANSACTION -")
    feature_name = feature_name.replace("Trans", "Transaction -")
    split_name = feature_name.split("_")
    if len(split_name) > 1:
        return " ".join(
            [
                (
                    word.capitalize().replace("Nunique", "\#Unique")
                    if "id" not in word
                    else ""
                )
                for word in split_name
            ]
        )

    return feature_name[0].upper() + feature_name[1:]


color_real = "#b50827"
color_synthetic = "#3f53c6"

fig, (ax, ax_histy) = plt.subplots(
    1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [5, 1]}
)


disp = PartialDependenceDisplay.from_estimator(
    metric.models[0],
    metric.X,
    [pair],
    response_method="predict_proba",
    contour_kw={"cmap": cmap},
    ax=ax,
    n_jobs=-1,
    percentiles=(0.0, 1.0),
)

ax = disp.axes_[0, 0]
ax.scatter(
    metric.X.loc[metric.y == 0, pair[0]],
    metric.X.loc[metric.y == 0, pair[1]],
    s=10,
    alpha=0.1,
    label="synthetic",
    color=color_synthetic,
)
ax.scatter(
    metric.X.loc[metric.y == 1, pair[0]],
    metric.X.loc[metric.y == 1, pair[1]],
    s=10,
    alpha=0.1,
    label="real",
    color=color_real,
)
# ax.legend(loc='upper right')


ax.set_xlabel(prettyify_feature_name(pair[0]), fontsize=30)
ax.set_ylabel(prettyify_feature_name(pair[1]), fontsize=30)
ax.tick_params(axis="both", labelsize=23)  # Set font size for x and y ticks


clip = None  # (metric.X[pair[1]].min(), metric.X[pair[1]].max())
sns.kdeplot(
    y=metric.X.loc[metric.y == 0, pair[1]],
    ax=ax_histy,
    alpha=0.4,
    label="Synthetic",
    color=color_synthetic,
    clip=clip,
    legend=False,
    fill=True,
)
sns.kdeplot(
    y=metric.X.loc[metric.y == 1, pair[1]],
    ax=ax_histy,
    alpha=0.4,
    label="Real",
    color=color_real,
    clip=clip,
    legend=False,
    fill=True,
)

ax_histy.set_ylabel(None)
ax_histy.legend(fontsize=23)
ax_histy.set_yticks([])  # Hide the y-ticks on the histogram axes
ax_histy.set_xticks([])  # Hide the x-ticks on the histogram axes
ax_histy.set_xlabel("")
ax_histy.tick_params(axis="both", labelsize=23)  # Set font size for x and y ticks


# Adjust the positions to prevent overlap.  Make the marginal plot narrower and move the PDP plot to the left
# ax.set_position([0.3, 0.1, 0.65, 0.8])  # [left, bottom, width, height]
# ax_histy.set_position([, 0.1, .2, 1.])


ax_histy.grid(False)


# remove the top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax_histy.spines["top"].set_visible(False)
ax_histy.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax_histy.spines["left"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax_histy.spines["bottom"].set_visible(False)

# Reduce the space between the two subplots


plt.tight_layout()
plt.subplots_adjust(wspace=0.01)  # Adjust the width space between the two subplots
plt.show()

if not os.path.exists("results/figures"):
    os.makedirs("results/figures")

plt.savefig("results/figures/figure3b.png", dpi=300)
