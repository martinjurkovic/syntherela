import json
import os

import pandas as pd
from syntheval import SynthEval

from syntherela.data import load_tables
from syntherela.metadata import Metadata


def eval_mia(
    syn_data,
    real_data,
    test_data,
    metadata,
):
    """Adapted from
    https://github.com/jacobyhsi/TabRep/blob/main/eval/eval_privacy.py.
    """
    id_columns = metadata.get_column_names(sdtype="id")
    datetime_columns = metadata.get_column_names(sdtype="datetime")
    for df in (real_data, test_data, syn_data):
        for col in id_columns:
            if col in df.columns:
                df.drop(columns=col, inplace=True)
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # find all categorical columns
        cat_cols = df.select_dtypes(include=['category']).columns
        if len(cat_cols):
            # convert each one to its integer codes
            for col in cat_cols:
                df[col] = df[col].cat.codes
            print(
                f"Converted categorical columns to int codes: {list(cat_cols)}"
            )

    if len(syn_data) < len(test_data):
        test_data = test_data.sample(n=len(syn_data), random_state=42)
    S = SynthEval(real_data, holdout_dataframe=test_data)
    eval_df = S.evaluate(
        syn_data, None, "mia"
    )  # set the target column to the primary key of the table

    # Filter for rows with 'mia_recall' and 'mia_precision'
    filtered_rows = eval_df[eval_df['metric'].isin(
        ['mia_recall', 'mia_precision']
    )]

    # Extract values into variables
    mia_recall_val = filtered_rows.loc[filtered_rows['metric'] == 'mia_recall',
                                       'val'].values[0]
    mia_recall_err = filtered_rows.loc[filtered_rows['metric'] == 'mia_recall',
                                       'err'].values[0]
    mia_precision_val = filtered_rows.loc[filtered_rows['metric'] ==
                                          'mia_precision', 'val'].values[0]
    mia_precision_err = filtered_rows.loc[filtered_rows['metric'] ==
                                          'mia_precision', 'err'].values[0]

    # Print extracted variables

    print("mia_precision_val:", mia_precision_val)
    print("mia_precision_err:", mia_precision_err)
    print("mia_recall_val:", mia_recall_val)
    print("mia_recall_err:", mia_recall_err)

    return mia_precision_val, mia_precision_err, mia_recall_val, mia_recall_err


if __name__ == "__main__":

    os.makedirs("results/mia", exist_ok=True)

    table = "sessions"
    method = "MOSTLYAI"

    data_path = "data/original/airbnb-simplified_subsampled"
    metadata = Metadata().load_from_json(f"{data_path}/metadata.json")

    tables_real = load_tables(data_path, metadata)
    tables_test = load_tables("data/original/airbnb-dcr", metadata)

    metadata.validate_data(tables_real)
    metadata.validate_data(tables_test)

    if os.path.exists("results/mia/all_results.json"):
        with open("results/mia/all_results.json") as f:
            all_results = json.load(f)
    else:
        all_results = {}
    # Initialize the dictionary to store results
    methods = [
        'MOSTLYAI', 'RGCLD', 'CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'SDV',
        'SMOTE', "MARE"
    ]

    for method in methods:
        tables_syn = load_tables(
            f"data/synthetic/airbnb-simplified_subsampled/{method}/1/sample1",
            metadata
        )
        metadata.validate_data(tables_syn)
        if method not in all_results:
            all_results[method] = {}
        for table in metadata.get_tables():
            print(f"Evaluating {table} with method {method}")
            syn_data = tables_syn[table].copy()
            real_data = tables_real[table].copy()
            test_data = tables_test[table].copy()
            mia_precision_val, mia_precision_err, mia_recall_val, mia_recall_err = eval_mia(
                syn_data,
                real_data,
                test_data,
                metadata.get_table_meta(table, to_dict=False),
            )
            all_results[method][table] = {
                "precision": mia_precision_val,
                "precision_err": mia_precision_err,
                "recall": mia_recall_val,
                "recall_err": mia_recall_err
            }

            # Save results to JSON after each table evaluation
            with open("results/mia/all_results.json", "w") as f:
                json.dump(all_results, f, indent=4)

    print("\nFinal Results:")
    print(json.dumps(all_results, indent=4))
