import re
import json
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

run = 1
method_names_dict = {
    "SDV": "SDV",
    "RCTGAN": "RCTGAN",
    "REALTABFORMER": "REALTABF.",
    "MOSTLYAI": "MOSTLYAI",
    "GRETEL_ACTGAN": "GRE-ACTGAN",
    "GRETEL_LSTM": "GRE-LSTM",
    "CLAVADDPM": "CLAVADDPM",
}

baselines = {"walmart": 7697, "rossmann": 345, "airbnb": 0.5}

dataset_names = {"walmart": "Walmart", "rossmann": "Rossmann", "airbnb": "AirBnB"}


def df_to_latex(df):
    latex_table = df.to_latex(index=False, escape=False, column_format="clrrr")
    # replace all white spaces with one space
    table_latex = re.sub(" +", " ", latex_table)
    table_latex = table_latex.replace("\\\\\n\\multirow", "\\\\\n\\hline\n\\multirow")
    table_latex = (
        table_latex.replace(".00\\", "\\")
        .replace(".000\\", "\\")
        .replace(".000$", "$")
        .replace(".00$", "$")
    )
    return table_latex


def add_utility_results(df, dataset, data, round_score=False):
    model = "xgboost"
    score = np.abs(data[dataset]["SDV"][model]["real_score"])
    score_se = data[dataset]["SDV"][model]["real_score_se"]
    if round_score:
        score = np.round(score).astype(int)
        if score_se > 1:
            score_se = np.round(score_se).astype(int)
        score_se = np.round(score_se, 1)

    header_row = pd.DataFrame(
        {
            "Dataset": f"\multirow{{7}}{{*}}{{\\parbox{{1.3cm}}{{{dataset_names[dataset]}}}}}",
            "Method": "Real Data",
            "Score": f"${score:.2f}\pm{score_se:.3f}$ ({baselines[dataset]})",
            "Model Selection": "-",
            "Feature Selection": "-",
        },
        index=[0],
    )
    # add the row without append
    df = pd.concat([df, header_row], ignore_index=True)

    for method in data[dataset].keys():
        score = np.abs(data[dataset][method][model]["synthetic_score"])
        score_se = data[dataset][method][model]["synthetic_score_se"]
        if round_score:
            score = np.round(score).astype(int)
            if score_se > 1:
                score_se = np.round(score_se).astype(int)
            score_se = np.round(score_se, 1)

        model_selection = data[dataset][method]["spearman_mean"]
        model_selection_se = data[dataset][method]["spearman_se"]
        feature_selection = data[dataset][method]["feature_importance_spearman_mean"]
        feature_selection_se = data[dataset][method]["feature_importance_spearman_se"]

        new_row = pd.DataFrame(
            {
                "Dataset": "",
                "Method": method_names_dict[method],
                "Score": f"${score:.2f}\pm{score_se:.3f}$",
                "Model Selection": f"${model_selection:.2f}\pm{model_selection_se:.2f}$",
                "Feature Selection": f"${feature_selection:.2f}\pm{feature_selection_se:.2f}$",
            },
            index=[0],
        )
        df = pd.concat([df, new_row], ignore_index=True)
    return df


def add_rank_results(df, dataset, data, feature_importance=False):
    methods = list(data[dataset].keys())
    method = methods[0]
    if feature_importance:
        key_spearman = "feature_importance_spearman"
        key_kendall = "feature_importance_tau"
        key_weighted = "feature_importance_weighted"
    else:
        key_spearman = "spearman"
        key_kendall = "kendall"
        key_weighted = "weighted"
    spearman = data[dataset][method][f"{key_spearman}_mean"]
    spearman_se = data[dataset][method][f"{key_spearman}_se"]
    kendall_Kendall = data[dataset][method][f"{key_kendall}_mean"]
    kendall_Kendall_se = data[dataset][method][f"{key_kendall}_se"]
    weighted = data[dataset][method][f"{key_weighted}_mean"]
    weighted_se = data[dataset][method][f"{key_weighted}_se"]

    header_row = pd.DataFrame(
        {
            "Dataset": f"\multirow{{7}}{{*}}{{\\parbox{{1.3cm}}{{{dataset_names[dataset]}}}}}",
            "Method": method_names_dict[method],
            "Spearman": f"${spearman:.2f}\pm{spearman_se:.2f}$",
            "Kendall": f"${kendall_Kendall:.2f}\pm{kendall_Kendall_se:.2f}$",
            "Weighted": f"${weighted:.2f}\pm{weighted_se:.2f}$",
        },
        index=[0],
    )
    df = pd.concat([df, header_row], ignore_index=True)

    for method in methods[1:]:
        spearman = data[dataset][method][f"{key_spearman}_mean"]
        spearman_se = data[dataset][method][f"{key_spearman}_se"]
        kendall_Kendall = data[dataset][method][f"{key_kendall}_mean"]
        kendall_Kendall_se = data[dataset][method][f"{key_kendall}_se"]
        weighted = data[dataset][method][f"{key_weighted}_mean"]
        weighted_se = data[dataset][method][f"{key_weighted}_se"]

        new_row = pd.DataFrame(
            {
                "Dataset": "",
                "Method": method_names_dict[method],
                "Spearman": f"${spearman:.2f}\pm{spearman_se:.2f}$",
                "Kendall": f"${kendall_Kendall:.2f}\pm{kendall_Kendall_se:.2f}$",
                "Weighted": f"${weighted:.2f}\pm{weighted_se:.2f}$",
            },
            index=[0],
        )
        df = pd.concat([df, new_row], ignore_index=True)
    return df


df = pd.DataFrame(
    columns=["Dataset", "Method", "Score", "Model Selection", "Feature Selection"]
)
df_model = pd.DataFrame(
    columns=["Dataset", "Method", "Spearman", "Kendall", "Weighted"]
)
df_feature = pd.DataFrame(
    columns=["Dataset", "Method", "Spearman", "Kendall", "Weighted"]
)

for dataset in ["airbnb", "rossmann", "walmart"]:
    with open(f"results/mle_{dataset}_{run}_0.json") as f:
        results = json.load(f)
    round_score = dataset != "airbnb"
    df = add_utility_results(df, dataset, results, round_score)
    df_model = add_rank_results(df_model, dataset, results)
    df_feature = add_rank_results(df_feature, dataset, results, feature_importance=True)

# Save the tables


print("Saving Table 2 (best results were bolded manually).")
table_latex = df_to_latex(df)
with open("results/tables/table2.tex", "w") as f:
    f.write(table_latex)

print("Saving Table 7 (best results were bolded manually).")
table_latex = df_to_latex(df_model)
with open("results/tables/table7.tex", "w") as f:
    f.write(table_latex)

print("Saving Table 8 (best results were bolded manually).")
table_latex = df_to_latex(df_feature)
with open("results/tables/table8.tex", "w") as f:
    f.write(table_latex)
