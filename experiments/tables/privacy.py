import json

import pandas as pd
import seaborn as sns

from syntherela.metadata import Metadata

sns.set(style="whitegrid")

metadata = Metadata(
).load_from_json("data/original/airbnb-simplified_subsampled/metadata.json")

with open("results/dcr/all_results.json") as f:
    all_results = json.load(f)

results = pd.DataFrame(all_results).T
METHOD = '\\textbf{Method}'
results[METHOD] = results.index
for column in results.columns:
    if column == METHOD:
        continue
    results["\\textbf{" + column.title() + "}"] = results.pop(column).apply(
        lambda x: f"${x['score'] * 100: .2f}$" + " {\\tiny" +
        fr"$\pm{x['se'] * 100 : .2f}$" + "}"
    )

results.reset_index(drop=True, inplace=True)

methods = [
    'MOSTLYAI', 'RGCLD', 'CLAVADDPM', 'RCTGAN', 'REALTABFORMER', 'SDV', 'SMOTE'
]

model_names = {
    'CLAVADDPM': "ClavaDDPM",
    'RGCLD': "RGCLD",
    'MOSTLYAI': "TabularARGN",
    'RCTGAN': "RCTGAN",
    'REALTABFORMER': "REALTABF.",
    'SDV': "SDV",
    'SMOTE': "SMOTE",
}

# sort the results by methods
results = results[results[METHOD].isin(methods)]
# rename the methods
results[METHOD] = results[METHOD].map(model_names)

df_latex = results.to_latex(column_format="ccc", index=False)

with open("results/tables/table5.tex", "w") as f:
    f.write(df_latex)
