import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns

sns.set_theme()
rc("font", **{"family": "serif", "serif": ["Times"], "size": 30})
rc("text", usetex=True)

methods = ["SDV", "RCTGAN", "REALTABFORMER", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM", "ClavaDDPM"]


def prettify_method_name(method_name):
    if method_name == "REALTABFORMER":
        return "REALTABF."
    if method_name == "GRETEL_ACTGAN":
        return "G-ACTGAN"
    if method_name == "GRETEL_LSTM":
        return "G-LSTM"
    return method_name


# Load original data
metadata = Metadata().load_from_json(f"data/original/rossmann_subsampled/metadata.json")
tables = load_tables(f"data/original/rossmann_subsampled/", metadata)

tables, metadata = remove_sdv_columns(tables, metadata)

# Load synthetic data for all methods
all_tables = dict()
for method in methods:
    method_name = prettify_method_name(method)
    all_tables[method_name] = load_tables(
        f"data/synthetic/rossmann_subsampled/{method}/1/sample1", metadata
    )
    all_tables[method_name], metadata = remove_sdv_columns(
        all_tables[method_name], metadata, update_metadata=False
    )

COLORMAP = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

COLOR_DICT = {
    "REAL": COLORMAP[6],
    "MOSTLYAI": COLORMAP[8],
    "SDV": COLORMAP[0],
    "RCTGAN": COLORMAP[1],
    "REALTABF.": COLORMAP[2],
    "G-ACTGAN": COLORMAP[3],
    "G-LSTM": COLORMAP[5],
    "ClavaDDPM": COLORMAP[4],
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

all_tables = dict()
for method in methods:
    method_name = prettify_method_name(method)
    all_tables[method_name] = load_tables(
        f"data/synthetic/rossmann_subsampled/{method}/1/sample1", metadata
    )
    all_tables[method_name], metadata = remove_sdv_columns(
        all_tables[method_name], metadata, update_metadata=False
    )

table = "store"
column1 = "PromoInterval"
columns = [
    tables[table][column1],
]
for method in methods:
    method = prettify_method_name(method)
    columns.append(all_tables[method][table][column1])
data = pd.DataFrame(pd.concat(columns, axis=0)).astype("object")
data.fillna("missing", inplace=True)


# map the 'Jan,Apr,Jul,Oct' to 'J,A,O'; 'Feb,May,Aug,Nov' to 'F,M,A,N'; 'Mar,Jun,Sep,Dec' to 'M,J,S,D'
data[column1] = (
    data[column1]
    .astype("object")
    .map(
        {
            "Jan,Apr,Jul,Oct": "J-A-J-O",
            "Feb,May,Aug,Nov": "F-M-A-N",
            "Mar,Jun,Sept,Dec": "M-J-S-D",
            "missing": "missing",
        }
    )
)

# sort the data by the frequency of the values in the real data
column_order = (
    tables[table][column1]
    .astype("object")
    .fillna("missing")
    .map(
        {
            "Jan,Apr,Jul,Oct": "J-A-J-O",
            "Feb,May,Aug,Nov": "F-M-A-N",
            "Mar,Jun,Sept,Dec": "M-J-S-D",
            "missing": "missing",
        }
    )
    .value_counts()
    .index.tolist()
)
# map the column order to the data

data[column1] = pd.Categorical(data[column1], categories=column_order, ordered=True)

methods_column = ["REAL"] * len(tables[table])
for method in methods:
    method = prettify_method_name(method)
    methods_column = methods_column + [method] * len(all_tables[method][table])
data["Data"] = methods_column

sns.histplot(
    data=data.dropna(),
    x=column1,
    hue="Data",
    multiple="dodge",
    stat="density",
    common_norm=False,
    legend=True,
    ax=axes[0],
    shrink=0.8,
    palette=COLOR_DICT,
    alpha=0.8,
)

table = "historical"
column2 = "Customers"
columns = [
    tables[table][column2],
]
for method in methods:
    method = prettify_method_name(method)
    columns.append(all_tables[method][table][column2])
data = pd.DataFrame(pd.concat(columns, axis=0))

methods_column = ["REAL"] * len(tables[table])
for method in methods:
    method = prettify_method_name(method)
    methods_column = methods_column + [method] * len(all_tables[method][table])
data["Data"] = methods_column

data.dropna(inplace=True)
sns.kdeplot(
    data=data,
    x=column2,
    hue="Data",
    common_norm=False,
    fill=False,
    legend=True,
    ax=axes[1],
    clip=(0, data[column2].max()),
    palette=COLOR_DICT,
    alpha=0.8,
)

fig.suptitle = f"{table}"

axes[0].tick_params(axis="x", labelsize=16)
axes[0].tick_params(axis="y", labelsize=16)
axes[0].set_ylabel("Density", fontsize=20)
axes[0].set_xlabel("PromoInterval", fontsize=20)
plt.setp(axes[0].get_legend().get_texts(), fontsize="16")
plt.setp(axes[0].get_legend().get_title(), fontsize="16")

axes[1].tick_params(axis="x", labelsize=16)
axes[1].tick_params(axis="y", labelsize=16)
axes[1].set_ylabel("Density", fontsize=20)
axes[1].set_xlabel("Customers", fontsize=20)
plt.setp(axes[1].get_legend().get_texts(), fontsize="16")
plt.setp(axes[1].get_legend().get_title(), fontsize="16")

fig.tight_layout()

plt.savefig("results/figures/figure1.png", dpi=600)
