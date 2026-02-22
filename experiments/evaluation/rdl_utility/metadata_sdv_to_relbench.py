import json
import os
from pathlib import Path

DATASETS = [
    "rossmann_subsampled",
    "walmart_subsampled",
    "airbnb-simplified_subsampled",
    "f1_subsampled",
    "Berka_subsampled",
]


def convert_sdv_to_relbench_type(
    sdtype, column_name, computer_representation=None
):
    """Convert SDV data type to relbench data type."""
    # Basic mapping rules based on SDV types
    if sdtype == "datetime":
        return "timestamp"
    elif sdtype == "boolean":
        return "categorical"
    elif sdtype == "id":
        return "numerical"
    elif sdtype == "categorical":
        return "categorical"
    elif sdtype == "numerical":
        return "numerical"
    else:
        # Default fallback
        raise ValueError(f"Unknown SDV type: {sdtype}")


def convert_metadata(sdv_metadata):
    """Convert SDV metadata to relbench format."""
    relbench_metadata = {}

    for table_name, table_info in sdv_metadata["tables"].items():
        relbench_metadata[table_name] = {}

        for column_name, column_info in table_info["columns"].items():
            sdtype = column_info["sdtype"]
            computer_representation = column_info.get("computer_representation")

            relbench_type = convert_sdv_to_relbench_type(
                sdtype, column_name, computer_representation
            )
            relbench_metadata[table_name][column_name] = relbench_type

    return relbench_metadata


def main():
    """Convert SDV metadata to relbench format for all datasets."""
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")

        # Load SDV metadata
        sdv_metadata_path = f"data/original/{dataset}/metadata.json"

        if not os.path.exists(sdv_metadata_path):
            print(f"Warning: {sdv_metadata_path} not found, skipping...")
            continue

        with open(sdv_metadata_path) as f:
            sdv_metadata = json.load(f)

        # Convert to relbench format
        relbench_metadata = convert_metadata(sdv_metadata)

        if dataset == "f1_subsampled":
            pass

        # Create output directory
        output_dir = Path.home(
        ) / ".cache" / "relbench_examples" / dataset  # / "tasks" / task
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save relbench metadata
        output_path = output_dir / "stypes.json"
        with open(output_path, 'w') as f:
            json.dump(relbench_metadata, f, indent=2)

        print(f"Saved relbench metadata to: {output_path}")


if __name__ == "__main__":
    main()
