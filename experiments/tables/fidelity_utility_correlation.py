import json

import numpy as np
import pandas as pd
from tqdm import tqdm   
from scipy.stats import pearsonr

def read_utility_results(dataset, model='xgboost', run="1"):
    utility_model = []
    utility_feature = []
    utility_score = []

    dataset_name = dataset.split("-")[0].split("_")[0]
    with open(f"results/mle_{dataset_name}_{run}_0.json", "r") as f:
        results = json.load(f)[dataset_name]
    for method in ["SDV", "RCTGAN", "REALTABFORMER", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM", "CLAVADDPM"]:
        utility_model.append(results[method]["spearman_mean"])
        utility_feature.append(results[method]["feature_importance_spearman_mean"])
        utility_score.append(results[method][model]["synthetic_score"])
    utility_scores = np.array(utility_score) 
    # Normalize the utility scores so they are comparable between datasets
    utility_scores = (utility_scores - np.min(utility_scores)) / (np.max(utility_scores) - np.min(utility_scores))
    return utility_model, utility_feature, utility_scores.tolist()


def read_fidelity_results(dataset, model, metric, target_table=None, target_column=None, run="1"):
    results = []
    for method in ["SDV", "RCTGAN", "REALTABFORMER", "MOSTLYAI", "GRETEL_ACTGAN", "GRETEL_LSTM", "CLAVADDPM"]:
        with open(f"results/{run}/{dataset}_{method}_{run}_sample1.json", "r") as f:
            method_results = json.load(f)
        if 'Aggregation' in metric:
            if dataset == "rossmann_subsampled":
                root_table = "store"
            elif dataset == "airbnb-simplified_subsampled":
                root_table = "users"
            elif dataset == "walmart_subsampled":
                root_table = "stores"
            results.append(method_results['multi_table_metrics'][f'{metric}-{model}'][root_table]['accuracy'])
        elif 'SingleColumn' in metric:
            results.append(method_results['single_column_metrics'][f'{metric}-{model}'][target_table][target_column]['accuracy'])
        elif 'SingleTable' in metric:
            results.append(method_results['single_table_metrics'][f'{metric}-{model}'][target_table]['accuracy'])
    return results


fidelities_xgb = []
fidelities_lin = []
utilities = []
dataset_list = []
method_list = []
run_list = []
for dataset in ["rossmann_subsampled", "walmart_subsampled","airbnb-simplified_subsampled"]:
    if dataset == "rossmann_subsampled":
        target_table = "historical"
        target_column = "Customers"
    elif dataset == "airbnb-simplified_subsampled":
        target_table = "users"
        target_column = "country_destination"
    elif dataset == "walmart_subsampled":
        target_table = "depts"
        target_column = "Weekly_Sales"

    for run in range(3):
        run_id = str(run + 1)

        # Fidelity
        fidelity_xgb = read_fidelity_results(dataset, "XGBClassifier", "AggregationDetection", run=run_id)
        fidelity_lin = read_fidelity_results(dataset, "LogisticRegression", "AggregationDetection", target_table, run=run_id)
                
        utility_scores = []
        for model in ["xgboost", "linear", "random_forest", "decision_tree", "knn", "svr", "mlp", "svc", "gaussian_nb"]:
            if dataset == 'airbnb-simplified_subsampled':
                if model =='svr':
                    continue
            else:
                if model == 'svc' or model == 'gaussian_nb':
                    continue

            # Utility
            try:
                utility_model, utility_feature, utility_score = read_utility_results(dataset, model, run=run_id)
            except FileNotFoundError:
                print(f"Run {run_id} not found for {dataset}")
                utility_model, utility_feature, utility_score = read_utility_results(dataset, model)
            
            utility_scores.append(utility_score)
            
        utility_score = np.array(utility_scores).mean(axis=0).tolist()
        fidelities_xgb.extend(fidelity_xgb)
        fidelities_lin.extend(fidelity_lin)
        utilities.extend(utility_score)
        dataset_list += [dataset] * len(fidelity_xgb)
        run_list += [run_id] * len(fidelity_xgb)

utilities = np.array(utilities)
fidelities_xgb = np.array(fidelities_xgb)
fidelities_lin = np.array(fidelities_lin)
dataset_list = np.array(dataset_list)
run_list = np.array(run_list)


def bootstrap_correlations(utilities, fidelities_xgb, fidelities_lin, dataset_list, datasets, m=10000):
    xgb_corr = []
    lin_corr = []
    xgb_lin_diff = []
    for i in tqdm(range(m)):
        indices = np.where(np.isin(dataset_list, datasets))[0]
        np.random.seed(i)
        indices = np.random.choice(indices, len(indices), replace=True)
        utility_score = utilities[indices]
        fidelity_xgb = fidelities_xgb[indices]
        fidelity_lin = fidelities_lin[indices]

        corr_xgb = pearsonr(utility_score, fidelity_xgb)[0]
        corr_lin = pearsonr(utility_score, fidelity_lin)[0]

        xgb_corr.append(corr_xgb)
        lin_corr.append(corr_lin)
        xgb_lin_diff.append(corr_xgb - corr_lin)
    return xgb_corr, lin_corr, xgb_lin_diff


datasets = ["rossmann_subsampled", "walmart_subsampled","airbnb-simplified_subsampled"]
results = {}
for dataset in datasets:
    xgb_corr, lin_corr, xgb_lin_diff = bootstrap_correlations(utilities, fidelities_xgb, fidelities_lin, dataset_list, [dataset])
    results[dataset] = {
        "xgb_corr": xgb_corr,
        "lin_corr": lin_corr,
        "xgb_lin_diff": xgb_lin_diff
    }
xgb_corr, lin_corr, xgb_lin_diff = bootstrap_correlations(utilities, fidelities_xgb, fidelities_lin, dataset_list, ["rossmann_subsampled", "walmart_subsampled","airbnb-simplified_subsampled"])
results['all'] = {
    "xgb_corr": xgb_corr,
    "lin_corr": lin_corr,
    "xgb_lin_diff": xgb_lin_diff
}


dataset_names = {
    "rossmann_subsampled": "Rossmann",
    "walmart_subsampled": "Walmart",
    "airbnb-simplified_subsampled": "Airbnb",
    "all": "Total"
}

def format_result(result, ci=False):
    import math
    se = np.std(result) / np.sqrt(len(result))
    mean = np.mean(result)
    se_digit = abs(int(math.log10(abs(se + 1e-8))))
    if ci:
        q = np.quantile(result, [0.05, 0.95])
        return f"${mean.round(se_digit)} ({q[0].round(se_digit)}, {q[1].round(se_digit)})$"

    return f"${mean.round(se_digit)}$".replace("-0.0 ", "0 ")

results_df = pd.DataFrame(columns=["Dataset", "$\\rho(DDA_{XGB}, U)$", "$\\rho(LD, U)$", "$\\rho_{DDA} - \\rho_{LD}$"])
for dataset, result in results.items():
    diff = np.array(result['xgb_corr']) - np.array(result['lin_corr'])
    ddxgb = format_result(result['xgb_corr'])
    ld = format_result(result['lin_corr'])
    diff = format_result(diff, ci=True)
    results_df.loc[len(results_df)] = [dataset_names[dataset], ddxgb, ld, diff]

results_df.to_latex("results/tables/table6.tex", index=False, column_format="lccc")