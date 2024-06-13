import warnings

import xgboost as xgb
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

from syntherela.metadata import Metadata
from syntherela.metrics.multi_table.detection import AggregationDetection
from syntherela.data import load_tables, remove_sdv_columns

warnings.filterwarnings("ignore")
sns.set_theme()
rc("font", **{"family": "serif", "serif": ["Times"], "size": 30})
rc("text", usetex=True)


def reproduce_figure(
    tables, tables_synthetic, metadata, target_table, feature, figure_name
):
    # Compute the metric
    xgb_cls = xgb.XGBClassifier
    xgb_args = {
        "seed": 0,
    }

    metric = AggregationDetection(
        classifier_cls=xgb_cls, classifier_args=xgb_args, random_state=42
    )

    for table in tables.keys():
        tables_synthetic[table] = tables_synthetic[table][tables[table].columns]

    metric.run(
        tables,
        tables_synthetic,
        metadata=metadata,
        target_table=target_table,
    )

    # Plot the feature importance

    metric.plot_partial_dependence(feature, seed=0)
    plt.savefig(
        f"results/figures/figure6{figure_name}.png", bbox_inches="tight", dpi=600
    )


dataset_name = "imdb_MovieLens_v1"
method = "GRETEL_ACTGAN"

metadata = Metadata().load_from_json(f"data/original/{dataset_name}/metadata.json")

tables = load_tables(f"data/original/{dataset_name}/", metadata)
tables_synthetic = load_tables(
    f"data/synthetic/{dataset_name}/{method}/1/sample1", metadata
)

tables, metadata = remove_sdv_columns(tables, metadata)
tables_synthetic, metadata = remove_sdv_columns(
    tables_synthetic, metadata, update_metadata=False
)

## FIGURE 6 (a)
feature = "movies2actors_movieid_cast_num_nunique"
reproduce_figure(tables, tables_synthetic, metadata, "movies", feature, "a")
## FIGURE 6 (b)
feature = "u2base_movieid_rating_mean"
reproduce_figure(tables, tables_synthetic, metadata, "movies", feature, "b")
