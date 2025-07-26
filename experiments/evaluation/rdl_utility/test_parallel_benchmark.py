import os
import subprocess
import time
from dotenv import load_dotenv

load_dotenv()

"""
Test script for parallel benchmark setup

Runs a quick test of each dataset on its assigned GPU with minimal parameters
to verify the setup works before running the full benchmark.
"""

PROJECT_PATH = os.getenv("PROJECT_PATH")

# Dataset to GPU mapping (same as master_benchmark.py)
DATASET_GPU_MAPPING = {
    "rossmann_subsampled": "cuda:9",
    "walmart_subsampled": "cuda:7", 
    "airbnb-simplified_subsampled": "cuda:6",
    "f1_subsampled": "cuda:5",
    "Berka_subsampled": "cuda:4",
}

def test_single_run(dataset, gpu_device):
    """Test a single quick run for a dataset"""
    print(f"Testing {dataset} on {gpu_device}...")
    
    # Base command arguments
    cmd_args = [
        "python", 
        "experiments/evaluation/rdl_utility/run_gnn.py",
        "--dataset", dataset,
        "--gnn_architecture", "hetero-graphsage",
        "--method", "ORIGINAL",
        "--run_id", "1",
        "--torch_device", gpu_device,
        "--epochs", "1",
        "--max_steps_per_epoch", "2",
        "--task", "autocomplete"
    ]
    
    # Add dataset-specific arguments
    if dataset == "rossmann_subsampled":
        cmd_args.extend(["--entity_table", "historical", "--target_col", "Customers", "--task_type", "REGRESSION"])
    elif dataset == "walmart_subsampled":
        cmd_args.extend(["--entity_table", "depts", "--target_col", "Weekly_Sales", "--task_type", "REGRESSION"])
    elif dataset == "airbnb-simplified_subsampled":
        cmd_args.extend(["--entity_table", "users", "--target_col", "country_destination", "--task_type", "BINARY_CLASSIFICATION"])
    elif dataset == "f1_subsampled":
        cmd_args.extend(["--task", "driver-top3", "--task_type", "BINARY_CLASSIFICATION"])
    elif dataset == "Berka_subsampled":
        cmd_args.extend(["--entity_table", "loan", "--target_col", "status", "--task_type", "BINARY_CLASSIFICATION"])
    
    command = cmd_args
    
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ {dataset} test passed ({duration:.1f}s)")
        return True
    else:
        print(f"✗ {dataset} test failed")
        print(f"Error: {result.stderr}")
        return False

def main():
    print("=== Testing Parallel Benchmark Setup ===")
    print("Running quick tests on each GPU...")
    print("="*50)
    
    success_count = 0
    total_count = len(DATASET_GPU_MAPPING)
    
    for dataset, gpu in DATASET_GPU_MAPPING.items():
        if test_single_run(dataset, gpu):
            success_count += 1
    
    print("="*50)
    print(f"Test Results: {success_count}/{total_count} passed")
    
    if success_count == total_count:
        print("✓ All tests passed! Ready to run full benchmark with master_benchmark.py")
    else:
        print("✗ Some tests failed. Check the errors above before running full benchmark.")
    
    print("="*50)

if __name__ == "__main__":
    main() 