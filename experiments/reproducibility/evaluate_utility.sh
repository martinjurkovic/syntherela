#!/bin/bash

python experiments/evaluation/rdl_utility/run_utility_benchmark.py
python experiments/evaluation/rdl_utility/run_singletable_benchmark.py --singletable_model singletable
python experiments/evaluation/rdl_utility/run_singletable_benchmark.py --singletable_model singletable_dfs
