#!/bin/bash

mkdir -p results/tables
echo "Writing fidelity tables (1, 9, 10)"
python experiments/tables/fidelity.py
echo "Writing utility tables (2, 7, 8)"
python experiments/tables/utility.py
echo "Writing fidelity - utility correlation table (6)"
python experiments/tables/fidelity_utility_correlation.py
echo "Tables can be found in results/tables"
