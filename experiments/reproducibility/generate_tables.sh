#!/bin/bash

mkdir -p results/tables
echo "Writing fidelity tables."
python experiments/tables/fidelity.py
echo "Writing utility table."
python experiments/tables/gnn_utility.py
echo "Writing rdl utility table."
python experiments/tables/rdl_utility.py
echo "Writing privacy table."
python experiments/tables/privacy.py
echo "Tables can be found in results/tables"
