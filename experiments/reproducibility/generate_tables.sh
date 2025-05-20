#!/bin/bash

mkdir -p results/tables
echo "Writing fidelity tables (2, 3, 7)."
python experiments/tables/fidelity.py
echo "Writing utility table (4)."
python experiments/tables/gnn_utility.py
echo "Writing privacy table (5)."
python experiments/tables/privacy.py
echo "Writing fidelity - utility correlation table (8)"
python experiments/tables/fidelity_utility_correlation.py
echo "Tables can be found in results/tables"
