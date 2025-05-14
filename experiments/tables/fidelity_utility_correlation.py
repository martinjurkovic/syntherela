import json
import os

import numpy as np
from dotenv import load_dotenv
from scipy.stats import spearmanr, kendalltau

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH")

datasets_methods = {
    "rossmann_subsampled": [
        "CLAVADDPM",
        "MOSTLYAI",
        "RCTGAN",
        "REALTABFORMER",
        "RGCLD",
        "SDV",
    ],
    "walmart_subsampled": [
        "CLAVADDPM",
        "MOSTLYAI",
        "RCTGAN",
        "REALTABFORMER",
        "RGCLD",
        "SDV",
    ],
    "airbnb-simplified_subsampled": [
        "CLAVADDPM",
        "MOSTLYAI",
        "RCTGAN",
        "RGCLD",
        "SDV",
    ],
    "Berka_subsampled": [
        "CLAVADDPM",
        "MOSTLYAI",
        "RGCLD",
    ],
    "f1_subsampled": [
        "CLAVADDPM",
        "MOSTLYAI",
        "RCTGAN",
        "RGCLD",
        "SDV",
    ],
}

dataset_rdl_utility_target_table = {
    "rossmann_subsampled": "store",
    "walmart_subsampled": "stores",
    "f1_subsampled": "drivers",
    "airbnb-simplified_subsampled": "users",
    "Berka_subsampled": "account",
}

datasets_evaluation_type = {
    "rossmann_subsampled": "mae",
    "walmart_subsampled": "mae",
    "f1_subsampled": "roc_auc",
    "airbnb-simplified_subsampled": "roc_auc",
    "Berka_subsampled": "roc_auc",
}

dataset_rename = {
    "f1_subsampled": "F1",
    "Berka_subsampled": "Berka",
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
}

rdl_utility_results = json.load(
    open(os.path.join(PROJECT_PATH, "results/gnn_utility_results.json"))
)

average_rdl_utility_results = {}
for dataset, methods in datasets_methods.items():
    evaluation_type = datasets_evaluation_type[dataset]
    for method in methods + ["ORIGINAL"]:
        tmp_scores = []
        for run in range(1, 4):
            tmp_scores.append(
                rdl_utility_results[dataset][method][str(run)][evaluation_type]
            )
        average_rdl_utility_results.setdefault(dataset, {})[method] = np.mean(
            tmp_scores
        )

c2st_results = {}
one_hop_results = {}
cardinality_results = {}
for dataset, methods in datasets_methods.items():
    for method in methods:
        run_tmp_results = []
        run_one_hop_results = []
        run_cardinality_results = []
        for run in range(1, 4):
            tmp_results_file = json.load(
                open(
                    os.path.join(
                        PROJECT_PATH,
                        f"results/{run}/{dataset}_{method}_{run}_sample1.json",
                    )
                )
            )
            # tmp_results = []
            # table_names = list(
            #     tmp_results_file["multi_table_metrics"][
            #         "AggregationDetection-XGBClassifier"
            #     ].keys()
            # )
            # for table_name in table_names:
            #     tmp_results.append(
            #         tmp_results_file["multi_table_metrics"][
            #             "AggregationDetection-XGBClassifier"
            #         ][table_name]["accuracy"]
            #     )
            # run_tmp_results.append(np.mean(tmp_results))
            run_tmp_results.append(
                tmp_results_file["multi_table_metrics"][
                    "AggregationDetection-XGBClassifier"
                ][dataset_rdl_utility_target_table[dataset]]["accuracy"]
            )
            run_one_hop_results.append(
                tmp_results_file["multi_table_metrics"]["Trends"]["k_hop_similarity"][
                    "1"
                ]["mean"]
            )
            tables = list(tmp_results_file["multi_table_metrics"]["CardinalityShapeSimilarity"].keys())
            run_cardinality_results.append(
                tmp_results_file["multi_table_metrics"]["CardinalityShapeSimilarity"][tables[0]]["pval"]
            )
        c2st_results.setdefault(dataset, {})[method] = np.mean(run_tmp_results)
        one_hop_results.setdefault(dataset, {})[method] = np.mean(run_one_hop_results)
        cardinality_results.setdefault(dataset, {})[method] = np.mean(run_cardinality_results)

print(average_rdl_utility_results)
print("\n----------------------------------\n")
print(c2st_results)
print("\n----------------------------------\n")
print(one_hop_results)

# Calculate correlations
correlations_final = {}

# Iterate over datasets defined in datasets_methods
for dataset in datasets_methods.keys():
    # Initialize results for this dataset
    dataset_corr_results = {
        "spearman_c2st_rdl": np.nan,
        "p_value_spearman_c2st_rdl": np.nan,
        "kendall_c2st_rdl": np.nan,
        "p_value_kendall_c2st_rdl": np.nan,
        "spearman_onehop_rdl": np.nan,
        "p_value_spearman_onehop_rdl": np.nan,
        "kendall_onehop_rdl": np.nan,
        "p_value_kendall_onehop_rdl": np.nan,
        "spearman_cardinality_rdl": np.nan,
        "p_value_spearman_cardinality_rdl": np.nan,
        "kendall_cardinality_rdl": np.nan,
        "p_value_kendall_cardinality_rdl": np.nan,
        "message": None,
    }

    # Basic checks for data presence
    if dataset not in average_rdl_utility_results:
        dataset_corr_results["message"] = "Dataset not in average_rdl_utility_results"
        correlations_final[dataset] = dataset_corr_results
        continue
    if dataset not in c2st_results:  # Needed for common_methods basis
        dataset_corr_results["message"] = (
            "Dataset not in c2st_results (needed for common methods)"
        )
        correlations_final[dataset] = dataset_corr_results
        continue
    # one_hop_results check will be done before its specific correlation

    # Identify common methods (synthesizers present in both C2ST and RDL)
    c2st_dataset_methods = c2st_results[dataset].keys()
    rdl_dataset_methods = average_rdl_utility_results[dataset].keys()
    common_methods = sorted(
        [
            m
            for m in c2st_dataset_methods
            if m in rdl_dataset_methods and m != "ORIGINAL"
        ]
    )

    if len(common_methods) < 2:
        dataset_corr_results["message"] = (
            f"Only {len(common_methods)} common method(s) between C2ST & RDL, need at least 2 for correlation"
        )
        correlations_final[dataset] = dataset_corr_results
        continue

    # Prepare RDL Utility scores for ranking (higher value = better performance)
    rdl_scores_for_ranking = []
    evaluation_type = datasets_evaluation_type[dataset]
    for method in common_methods:
        method_rdl_score = average_rdl_utility_results[dataset][method]
        if evaluation_type == "mae":
            rdl_scores_for_ranking.append(-method_rdl_score)
        elif evaluation_type == "roc_auc":
            rdl_scores_for_ranking.append(method_rdl_score)
        else:  # Should not happen if datasets_evaluation_type is comprehensive
            rdl_scores_for_ranking.append(
                np.nan
            )  # Fallback, will likely cause NaN in correlation

    # --- C2ST vs RDL Correlations ---
    c2st_scores_for_ranking = [
        -c2st_results[dataset][method] for method in common_methods
    ]
    try:
        s_corr, s_p = spearmanr(c2st_scores_for_ranking, rdl_scores_for_ranking)
        k_corr, k_p = kendalltau(c2st_scores_for_ranking, rdl_scores_for_ranking)
        dataset_corr_results["spearman_c2st_rdl"] = s_corr
        dataset_corr_results["p_value_spearman_c2st_rdl"] = s_p
        dataset_corr_results["kendall_c2st_rdl"] = k_corr
        dataset_corr_results["p_value_kendall_c2st_rdl"] = k_p
    except Exception as e:
        error_msg = f" C2ST-RDL corr error: {str(e)};"
        dataset_corr_results["message"] = (
            (dataset_corr_results["message"] + error_msg)
            if dataset_corr_results["message"]
            else error_msg
        )

    # --- Cardinality vs RDL Correlations ---
    if dataset not in cardinality_results:
        error_msg = " Cardinality data missing for dataset;"
        dataset_corr_results["message"] = (
            (dataset_corr_results["message"] + error_msg)
            if dataset_corr_results["message"]
            else error_msg
        )
    else:
        missing_methods_in_cardinality = [
            m for m in common_methods if m not in cardinality_results[dataset]
        ]
        if missing_methods_in_cardinality:
            error_msg = f" Methods {missing_methods_in_cardinality} from common_methods not in cardinality_results[{dataset}];"
            dataset_corr_results["message"] = (
                (dataset_corr_results["message"] + error_msg)
                if dataset_corr_results["message"]
                else error_msg
            )
        else:
            cardinality_scores_for_ranking = [
                cardinality_results[dataset][method] for method in common_methods
            ] # Higher p-value is better for similarity
            try:
                s_corr, s_p = spearmanr(cardinality_scores_for_ranking, rdl_scores_for_ranking)
                k_corr, k_p = kendalltau(cardinality_scores_for_ranking, rdl_scores_for_ranking)
                dataset_corr_results["spearman_cardinality_rdl"] = s_corr
                dataset_corr_results["p_value_spearman_cardinality_rdl"] = s_p
                dataset_corr_results["kendall_cardinality_rdl"] = k_corr
                dataset_corr_results["p_value_kendall_cardinality_rdl"] = k_p
            except Exception as e:
                error_msg = f" Cardinality-RDL corr error: {str(e)};"
                dataset_corr_results["message"] = (
                    (dataset_corr_results["message"] + error_msg)
                    if dataset_corr_results["message"]
                    else error_msg
                )

    # --- OneHop vs RDL Correlations ---
    if dataset not in one_hop_results:
        error_msg = " OneHop data missing for dataset;"
        dataset_corr_results["message"] = (
            (dataset_corr_results["message"] + error_msg)
            if dataset_corr_results["message"]
            else error_msg
        )
    else:
        # Ensure all common_methods are in this dataset's one_hop_results
        missing_methods_in_onehop = [
            m for m in common_methods if m not in one_hop_results[dataset]
        ]
        if missing_methods_in_onehop:
            error_msg = f" Methods {missing_methods_in_onehop} from common_methods not in one_hop_results[{dataset}];"
            dataset_corr_results["message"] = (
                (dataset_corr_results["message"] + error_msg)
                if dataset_corr_results["message"]
                else error_msg
            )
        else:
            one_hop_scores_for_ranking = [
                one_hop_results[dataset][method] for method in common_methods
            ]  # Higher is better
            try:
                s_corr, s_p = spearmanr(
                    one_hop_scores_for_ranking, rdl_scores_for_ranking
                )
                k_corr, k_p = kendalltau(
                    one_hop_scores_for_ranking, rdl_scores_for_ranking
                )
                dataset_corr_results["spearman_onehop_rdl"] = s_corr
                dataset_corr_results["p_value_spearman_onehop_rdl"] = s_p
                dataset_corr_results["kendall_onehop_rdl"] = k_corr
                dataset_corr_results["p_value_kendall_onehop_rdl"] = k_p
            except Exception as e:
                error_msg = f" OneHop-RDL corr error: {str(e)};"
                dataset_corr_results["message"] = (
                    (dataset_corr_results["message"] + error_msg)
                    if dataset_corr_results["message"]
                    else error_msg
                )

    correlations_final[dataset] = dataset_corr_results

print("\n\n--- Rank Correlations with RDL Utility (Console Output) ---")
print(
    "Metrics vs RDL Utility. For C2ST, lower is better (scores are negated for ranking)."
)
print("For OneHop similarity, higher is better (raw scores used for ranking).")
print(
    "For RDL Utility, MAE is negated (lower is better), AUC used directly (higher is better)."
)
print(
    "----------------------------------------------------------------------------------------------------"
)

for dataset, data in correlations_final.items():
    eval_type = datasets_evaluation_type.get(dataset, "N/A")
    print(f"\nDataset: {dataset} (RDL eval type: {eval_type})")

    if data.get("message"):
        print(f"  Note: {data['message']}")

    print(f"  C2ST vs RDL:")
    print(
        f"    Spearman Correlation: {data['spearman_c2st_rdl']:.4f}, P-value: {data['p_value_spearman_c2st_rdl']:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {data['kendall_c2st_rdl']:.4f}, P-value: {data['p_value_kendall_c2st_rdl']:.4f}"
    )

    print(f"  Cardinality vs RDL:")
    print(
        f"    Spearman Correlation: {data['spearman_cardinality_rdl']:.4f}, P-value: {data['p_value_spearman_cardinality_rdl']:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {data['kendall_cardinality_rdl']:.4f}, P-value: {data['p_value_kendall_cardinality_rdl']:.4f}"
    )

    print(f"  OneHop vs RDL:")
    print(
        f"    Spearman Correlation: {data['spearman_onehop_rdl']:.4f}, P-value: {data['p_value_spearman_onehop_rdl']:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {data['kendall_onehop_rdl']:.4f}, P-value: {data['p_value_kendall_onehop_rdl']:.4f}"
    )

# --- LaTeX Table Generation (Combined Table) ---
print("\n\n--- LaTeX Table Output (Combined) ---")

print("\n\n% Kendall Tau Rank Correlations with RDL Utility")
print("\\begin{table}[h!]")
print("\\centering")
print("\\caption{Kendall Tau Rank Correlations of Fidelity Metrics with RDL Utility}")
print("\\begin{tabular}{cccc}")
print("\\toprule")
print("Dataset & C2ST-Agg vs RDL & Cardinality vs RDL & 1-HOP vs RDL \\\\")
print("\\midrule")

for dataset_name in datasets_methods.keys():  # Use defined order
    data = correlations_final.get(dataset_name, {})

    k_c2st = (
        f"{data.get('kendall_c2st_rdl', np.nan):.3f}"
        if not np.isnan(data.get("kendall_c2st_rdl", np.nan))
        else "NaN"
    )
    k_cardinality = (
        f"{data.get('kendall_cardinality_rdl', np.nan):.3f}"
        if not np.isnan(data.get("kendall_cardinality_rdl", np.nan))
        else "NaN"
    )
    k_onehop = (
        f"{data.get('kendall_onehop_rdl', np.nan):.3f}"
        if not np.isnan(data.get("kendall_onehop_rdl", np.nan))
        else "NaN"
    )

    display_name = dataset_rename.get(dataset_name, dataset_name)
    dataset_display_name = display_name.replace("_", "\\_")

    note = ""
    if data.get("message"):
        if (k_c2st == "NaN") and (
            "C2ST" in data["message"] or "common method" in data["message"]
        ):
            note += " (C2ST Issue)"
        if (k_cardinality == "NaN") and (
            "Cardinality" in data["message"] or "common method" in data["message"]
        ):
            if not (note and "common method" in data["message"]): # Avoid duplicate (Low N) notes if common
                 note += " (Card. Issue)"
        if (k_onehop == "NaN") and (
            "OneHop" in data["message"] or "common method" in data["message"]
        ):
            if not (note and "common method" in data["message"]):
                 note += " (1Hop Issue)"

        # General low N note if all are NaN due to common method count
        if (k_c2st == "NaN" and k_cardinality == "NaN" and k_onehop == "NaN" and
            "common method" in data.get("message", "")):
            note = " (Low N)"
        elif not note and data["message"]:  # Generic message if no specific parsing and note is empty
            note = " (Issue)"

    # Print values in the new order: C2ST, Cardinality, OneHop
    print(
        f"{dataset_display_name}{note} & {k_c2st} & {k_cardinality} & {k_onehop} \\\\"
    )

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:kendall_correlations_rdl}")
print("\\end{table}")
