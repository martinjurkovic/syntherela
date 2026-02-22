import json
import os

import numpy as np
from scipy.stats import spearmanr, kendalltau, sem

PROJECT_PATH = __file__.split("experiments")[0]
NUM_RUNS = 3

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
    open(os.path.join(PROJECT_PATH, "results/gnn_utility_results.json")))

per_run_rdl_utility_results = {}
for dataset, methods in datasets_methods.items():
    evaluation_type = datasets_evaluation_type[dataset]
    for method in methods + ["ORIGINAL"]:
        tmp_scores = []
        for run in range(1, NUM_RUNS + 1):
            try:
                score = rdl_utility_results[dataset][method][str(
                    run)][evaluation_type]
                tmp_scores.append(score)
            except KeyError:
                tmp_scores.append(np.nan)

        per_run_rdl_utility_results.setdefault(dataset,
                                               {}).setdefault(method, {})
        for i, score in enumerate(tmp_scores):
            if len(tmp_scores) == NUM_RUNS:
                per_run_rdl_utility_results[dataset][method][i] = score
            else:
                per_run_rdl_utility_results[dataset][method][i] = np.nan

per_run_c2st_results = {}
per_run_one_hop_results = {}
per_run_cardinality_results = {}

for dataset, methods in datasets_methods.items():
    for method in methods:
        run_c2st_tmp_scores = []
        run_one_hop_tmp_scores = []
        run_cardinality_tmp_scores = []

        for run_idx in range(NUM_RUNS):
            actual_run_number = run_idx + 1
            try:
                tmp_results_file = json.load(
                    open(
                        os.path.join(
                            PROJECT_PATH,
                            f"results/{actual_run_number}/{dataset}_{method}_{actual_run_number}_sample1.json",
                        )))
                run_c2st_tmp_scores.append(
                    tmp_results_file["multi_table_metrics"]
                    ["AggregationDetection-XGBClassifier"][
                        dataset_rdl_utility_target_table[dataset]]["accuracy"])
                run_one_hop_tmp_scores.append(
                    tmp_results_file["multi_table_metrics"]["Trends"]
                    ["k_hop_similarity"]["1"]["mean"])
                tables = list(tmp_results_file["multi_table_metrics"]
                              ["CardinalityShapeSimilarity"].keys())
                run_cardinality_tmp_scores.append(
                    tmp_results_file["multi_table_metrics"]
                    ["CardinalityShapeSimilarity"][tables[0]]["pval"])
            except FileNotFoundError:
                run_c2st_tmp_scores.append(np.nan)
                run_one_hop_tmp_scores.append(np.nan)
                run_cardinality_tmp_scores.append(np.nan)
            except KeyError:
                run_c2st_tmp_scores.append(np.nan)
                run_one_hop_tmp_scores.append(np.nan)
                run_cardinality_tmp_scores.append(np.nan)

        per_run_c2st_results.setdefault(dataset, {}).setdefault(method, {})
        for i, score in enumerate(run_c2st_tmp_scores):
            per_run_c2st_results[dataset][method][i] = score

        per_run_one_hop_results.setdefault(dataset, {}).setdefault(method, {})
        for i, score in enumerate(run_one_hop_tmp_scores):
            per_run_one_hop_results[dataset][method][i] = score

        per_run_cardinality_results.setdefault(dataset,
                                               {}).setdefault(method, {})
        for i, score in enumerate(run_cardinality_tmp_scores):
            per_run_cardinality_results[dataset][method][i] = score

print("--- Per Run RDL Utility Results (Sample Check) ---")
if "rossmann_subsampled" in per_run_rdl_utility_results and "CLAVADDPM" in per_run_rdl_utility_results[
        "rossmann_subsampled"]:
    print(
        f"Rossmann CLAVADDPM RDL: {per_run_rdl_utility_results['rossmann_subsampled']['CLAVADDPM']}"
    )
else:
    print("Rossmann CLAVADDPM RDL data not found for sample check.")
print("\n----------------------------------\n")
print("--- Per Run C2ST Results (Sample Check) ---")
if "rossmann_subsampled" in per_run_c2st_results and "CLAVADDPM" in per_run_c2st_results[
        "rossmann_subsampled"]:
    print(
        f"Rossmann CLAVADDPM C2ST: {per_run_c2st_results['rossmann_subsampled']['CLAVADDPM']}"
    )
else:
    print("Rossmann CLAVADDPM C2ST data not found for sample check.")
print("\n----------------------------------\n")

correlations_final = {}

for dataset in datasets_methods.keys():
    dataset_messages = []

    all_runs_s_c2st_rdl, all_runs_p_s_c2st_rdl = [], []
    all_runs_k_c2st_rdl, all_runs_p_k_c2st_rdl = [], []
    all_runs_s_card_rdl, all_runs_p_s_card_rdl = [], []
    all_runs_k_card_rdl, all_runs_p_k_card_rdl = [], []
    all_runs_s_onehop_rdl, all_runs_p_s_onehop_rdl = [], []
    all_runs_k_onehop_rdl, all_runs_p_k_onehop_rdl = [], []

    base_methods = [
        m for m in datasets_methods.get(dataset, []) if m != "ORIGINAL"
    ]

    common_methods_c2st_rdl = [
        m for m in base_methods
        if m in per_run_c2st_results.get(dataset, {}) and \
           m in per_run_rdl_utility_results.get(dataset, {})
    ]
    common_methods_card_rdl = [
        m for m in base_methods
        if m in per_run_cardinality_results.get(dataset, {}) and \
           m in per_run_rdl_utility_results.get(dataset, {})
    ]
    common_methods_onehop_rdl = [
        m for m in base_methods
        if m in per_run_one_hop_results.get(dataset, {}) and \
           m in per_run_rdl_utility_results.get(dataset, {})
    ]

    evaluation_type = datasets_evaluation_type[dataset]

    for run_idx in range(NUM_RUNS):
        run_error_messages = []

        if len(common_methods_c2st_rdl) < 2:
            s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
            if run_idx == 0:
                run_error_messages.append(
                    f"C2ST-RDL: Not enough common methods ({len(common_methods_c2st_rdl)})"
                )
        else:
            rdl_scores_run = []
            for method in common_methods_c2st_rdl:
                score = per_run_rdl_utility_results[dataset][method].get(
                    run_idx, np.nan)
                if evaluation_type == "mae":
                    rdl_scores_run.append(
                        -score if not np.isnan(score) else np.nan)
                else:
                    rdl_scores_run.append(score)

            c2st_scores_run = [
                -per_run_c2st_results[dataset][method].get(run_idx, np.nan)
                for method in common_methods_c2st_rdl
            ]

            valid_pairs = [(c, r)
                           for c, r in zip(c2st_scores_run, rdl_scores_run)
                           if not np.isnan(c) and not np.isnan(r)]
            if len(valid_pairs) < 2:
                s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                run_error_messages.append(
                    f"Run {run_idx+1} C2ST-RDL: <2 valid pairs ({len(valid_pairs)})"
                )
            else:
                c2st_v, rdl_v = zip(*valid_pairs)
                try:
                    s_corr, s_p = spearmanr(c2st_v, rdl_v)
                    k_corr, k_p = kendalltau(c2st_v, rdl_v)
                except Exception as e:
                    s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                    run_error_messages.append(
                        f"Run {run_idx+1} C2ST-RDL corr error: {e}")
        all_runs_s_c2st_rdl.append(s_corr)
        all_runs_p_s_c2st_rdl.append(s_p)
        all_runs_k_c2st_rdl.append(k_corr)
        all_runs_p_k_c2st_rdl.append(k_p)

        if len(common_methods_card_rdl) < 2:
            s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
            if run_idx == 0:
                run_error_messages.append(
                    f"Card-RDL: Not enough common methods ({len(common_methods_card_rdl)})"
                )
        else:
            rdl_scores_run = []
            for method in common_methods_card_rdl:
                score = per_run_rdl_utility_results[dataset][method].get(
                    run_idx, np.nan)
                if evaluation_type == "mae":
                    rdl_scores_run.append(
                        -score if not np.isnan(score) else np.nan)
                else:
                    rdl_scores_run.append(score)

            card_scores_run = [
                per_run_cardinality_results[dataset][method].get(
                    run_idx, np.nan) for method in common_methods_card_rdl
            ]

            valid_pairs = [(c, r)
                           for c, r in zip(card_scores_run, rdl_scores_run)
                           if not np.isnan(c) and not np.isnan(r)]
            if len(valid_pairs) < 2:
                s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                run_error_messages.append(
                    f"Run {run_idx+1} Card-RDL: <2 valid pairs ({len(valid_pairs)})"
                )
            else:
                card_v, rdl_v = zip(*valid_pairs)
                try:
                    s_corr, s_p = spearmanr(card_v, rdl_v)
                    k_corr, k_p = kendalltau(card_v, rdl_v)
                except Exception as e:
                    s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                    run_error_messages.append(
                        f"Run {run_idx+1} Card-RDL corr error: {e}")
        all_runs_s_card_rdl.append(s_corr)
        all_runs_p_s_card_rdl.append(s_p)
        all_runs_k_card_rdl.append(k_corr)
        all_runs_p_k_card_rdl.append(k_p)

        if len(common_methods_onehop_rdl) < 2:
            s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
            if run_idx == 0:
                run_error_messages.append(
                    f"OneHop-RDL: Not enough common methods ({len(common_methods_onehop_rdl)})"
                )
        else:
            rdl_scores_run = []
            for method in common_methods_onehop_rdl:
                score = per_run_rdl_utility_results[dataset][method].get(
                    run_idx, np.nan)
                if evaluation_type == "mae":
                    rdl_scores_run.append(
                        -score if not np.isnan(score) else np.nan)
                else:
                    rdl_scores_run.append(score)

            onehop_scores_run = [
                per_run_one_hop_results[dataset][method].get(run_idx, np.nan)
                for method in common_methods_onehop_rdl
            ]

            valid_pairs = [(o, r)
                           for o, r in zip(onehop_scores_run, rdl_scores_run)
                           if not np.isnan(o) and not np.isnan(r)]
            if len(valid_pairs) < 2:
                s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                run_error_messages.append(
                    f"Run {run_idx+1} OneHop-RDL: <2 valid pairs ({len(valid_pairs)})"
                )
            else:
                onehop_v, rdl_v = zip(*valid_pairs)
                try:
                    s_corr, s_p = spearmanr(onehop_v, rdl_v)
                    k_corr, k_p = kendalltau(onehop_v, rdl_v)
                except Exception as e:
                    s_corr, s_p, k_corr, k_p = np.nan, np.nan, np.nan, np.nan
                    run_error_messages.append(
                        f"Run {run_idx+1} OneHop-RDL corr error: {e}")
        all_runs_s_onehop_rdl.append(s_corr)
        all_runs_p_s_onehop_rdl.append(s_p)
        all_runs_k_onehop_rdl.append(k_corr)
        all_runs_p_k_onehop_rdl.append(k_p)

        if run_error_messages:
            dataset_messages.extend(run_error_messages)

    agg_results = {
        "message":
        "; ".join(list(set(dataset_messages))) if dataset_messages else None
    }

    for prefix, values_list, p_values_list in [
        ("spearman_c2st_rdl", all_runs_s_c2st_rdl, all_runs_p_s_c2st_rdl),
        ("kendall_c2st_rdl", all_runs_k_c2st_rdl, all_runs_p_k_c2st_rdl),
        ("spearman_cardinality_rdl", all_runs_s_card_rdl,
         all_runs_p_s_card_rdl),
        ("kendall_cardinality_rdl", all_runs_k_card_rdl,
         all_runs_p_k_card_rdl),
        ("spearman_onehop_rdl", all_runs_s_onehop_rdl,
         all_runs_p_s_onehop_rdl),
        ("kendall_onehop_rdl", all_runs_k_onehop_rdl, all_runs_p_k_onehop_rdl),
    ]:
        valid_corrs = [v for v in values_list if not np.isnan(v)]
        agg_results[f"{prefix}_mean"] = np.nanmean(values_list)
        agg_results[f"{prefix}_stderr"] = sem(
            values_list, nan_policy='omit') if len(valid_corrs) > 0 else np.nan
        agg_results[f"{prefix}_n"] = len(valid_corrs)

        valid_p_values = [p for p in p_values_list if not np.isnan(p)]
        agg_results[f"p_value_{prefix}_mean"] = np.nanmean(p_values_list)
        agg_results[f"p_value_{prefix}_n"] = len(valid_p_values)

    correlations_final[dataset] = agg_results

print("\n\n--- Rank Correlations with RDL Utility (Console Output) ---")

for dataset, data in correlations_final.items():
    eval_type = datasets_evaluation_type.get(dataset, "N/A")
    print(f"\nDataset: {dataset} (RDL eval type: {eval_type})")

    if data.get("message"):
        print(f"  Note: {data['message']}")

    s_mean = data.get('spearman_c2st_rdl_mean', np.nan)
    s_stderr = data.get('spearman_c2st_rdl_stderr', np.nan)
    s_n = data.get('spearman_c2st_rdl_n', 0)
    s_p_mean = data.get('p_value_spearman_c2st_rdl_mean', np.nan)

    k_mean = data.get('kendall_c2st_rdl_mean', np.nan)
    k_stderr = data.get('kendall_c2st_rdl_stderr', np.nan)
    k_n = data.get('kendall_c2st_rdl_n', 0)
    k_p_mean = data.get('p_value_kendall_c2st_rdl_mean', np.nan)

    print(f"  C2ST vs RDL:")
    print(
        f"    Spearman Correlation: {s_mean:.4f} (SE: {s_stderr:.4f}, N: {s_n}), Avg P-value: {s_p_mean:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {k_mean:.4f} (SE: {k_stderr:.4f}, N: {k_n}), Avg P-value: {k_p_mean:.4f}"
    )

    s_mean = data.get('spearman_cardinality_rdl_mean', np.nan)
    s_stderr = data.get('spearman_cardinality_rdl_stderr', np.nan)
    s_n = data.get('spearman_cardinality_rdl_n', 0)
    s_p_mean = data.get('p_value_spearman_cardinality_rdl_mean', np.nan)

    k_mean = data.get('kendall_cardinality_rdl_mean', np.nan)
    k_stderr = data.get('kendall_cardinality_rdl_stderr', np.nan)
    k_n = data.get('kendall_cardinality_rdl_n', 0)
    k_p_mean = data.get('p_value_kendall_cardinality_rdl_mean', np.nan)

    print(f"  Cardinality vs RDL:")
    print(
        f"    Spearman Correlation: {s_mean:.4f} (SE: {s_stderr:.4f}, N: {s_n}), Avg P-value: {s_p_mean:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {k_mean:.4f} (SE: {k_stderr:.4f}, N: {k_n}), Avg P-value: {k_p_mean:.4f}"
    )

    s_mean = data.get('spearman_onehop_rdl_mean', np.nan)
    s_stderr = data.get('spearman_onehop_rdl_stderr', np.nan)
    s_n = data.get('spearman_onehop_rdl_n', 0)
    s_p_mean = data.get('p_value_spearman_onehop_rdl_mean', np.nan)

    k_mean = data.get('kendall_onehop_rdl_mean', np.nan)
    k_stderr = data.get('kendall_onehop_rdl_stderr', np.nan)
    k_n = data.get('kendall_onehop_rdl_n', 0)
    k_p_mean = data.get('p_value_kendall_onehop_rdl_mean', np.nan)

    print(f"  OneHop vs RDL:")
    print(
        f"    Spearman Correlation: {s_mean:.4f} (SE: {s_stderr:.4f}, N: {s_n}), Avg P-value: {s_p_mean:.4f}"
    )
    print(
        f"    Kendall Tau Correlation: {k_mean:.4f} (SE: {k_stderr:.4f}, N: {k_n}), Avg P-value: {k_p_mean:.4f}"
    )

print("\n\n--- LaTeX Table Output (Combined) ---")

print("\n\n% Kendall Tau Rank Correlations with RDL Utility (Mean \\pm SE)")
print("\\begin{table}[h!]")
print("\\centering")
print(
    "\\caption{Mean Kendall Tau Rank Correlations ($\\pm$ Standard Error) of Fidelity Metrics with RDL Utility over "
    + str(NUM_RUNS) + " runs.}")
print("\\begin{tabular}{lccc}")
print("\\toprule")
print("Dataset & C2ST-Agg vs RDL & Cardinality vs RDL & 1-HOP vs RDL \\\\")
print("\\midrule")


def format_latex_value(mean, stderr, n, num_total_runs):
    if n == 0 or np.isnan(mean):
        return "NaN"

    val_str = f"${mean:.3f}$"
    if not np.isnan(stderr) and n > 1:
        val_str += f"{{\\tiny$\pm${stderr:.3f}}}"

    if n < num_total_runs and n > 0:
        val_str += f" (N={n})"
    return val_str


for dataset_name in datasets_methods.keys():
    data = correlations_final.get(dataset_name, {})

    k_c2st_mean = data.get('kendall_c2st_rdl_mean', np.nan)
    k_c2st_stderr = data.get('kendall_c2st_rdl_stderr', np.nan)
    k_c2st_n = data.get('kendall_c2st_rdl_n', 0)

    k_card_mean = data.get('kendall_cardinality_rdl_mean', np.nan)
    k_card_stderr = data.get('kendall_cardinality_rdl_stderr', np.nan)
    k_card_n = data.get('kendall_cardinality_rdl_n', 0)

    k_onehop_mean = data.get('kendall_onehop_rdl_mean', np.nan)
    k_onehop_stderr = data.get('kendall_onehop_rdl_stderr', np.nan)
    k_onehop_n = data.get('kendall_onehop_rdl_n', 0)

    k_c2st_str = format_latex_value(k_c2st_mean, k_c2st_stderr, k_c2st_n,
                                    NUM_RUNS)
    k_card_str = format_latex_value(k_card_mean, k_card_stderr, k_card_n,
                                    NUM_RUNS)
    k_onehop_str = format_latex_value(k_onehop_mean, k_onehop_stderr,
                                      k_onehop_n, NUM_RUNS)

    display_name = dataset_rename.get(dataset_name, dataset_name)
    dataset_display_name = display_name.replace("_", "\\_")

    note_parts = []
    dataset_level_message = data.get("message", "")
    if dataset_level_message:
        if "C2ST-RDL: Not enough common methods" in dataset_level_message and k_c2st_n == 0:
            pass
        elif "Card-RDL: Not enough common methods" in dataset_level_message and k_card_n == 0:
            pass
        elif "OneHop-RDL: Not enough common methods" in dataset_level_message and k_onehop_n == 0:
            pass
        else:
            if k_c2st_n < NUM_RUNS or k_card_n < NUM_RUNS or k_onehop_n < NUM_RUNS:
                simplified_msg = dataset_level_message
                for i in range(1, NUM_RUNS + 1):
                    simplified_msg = simplified_msg.replace(
                        f"Run {i} C2ST-RDL: <2 valid pairs", "")
                    simplified_msg = simplified_msg.replace(
                        f"Run {i} Card-RDL: <2 valid pairs", "")
                    simplified_msg = simplified_msg.replace(
                        f"Run {i} OneHop-RDL: <2 valid pairs", "")
                simplified_msg = simplified_msg.replace(";;",
                                                        ";").strip().strip(";")
                if simplified_msg and simplified_msg != "None":
                    note_parts.append(simplified_msg.split(';')[0])

    if k_c2st_n == 0 and k_card_n == 0 and k_onehop_n == 0 and not note_parts:
        if "Not enough common methods" in dataset_level_message:
            note_parts.append("Low N")

    note_str = ""
    if note_parts:
        note_str = f" ({', '.join(list(set(note_parts)))})"

    print(
        f"{dataset_display_name}{note_str} & {k_c2st_str} & {k_card_str} & {k_onehop_str} \\\\"
    )

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:kendall_correlations_rdl_mean_se}")
print("\\end{table}")
