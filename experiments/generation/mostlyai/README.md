## Generating Data using MOSTLY AI

In this repository we use the [MostlyAI API](https://mostly.ai/docs/generators/configure/set-table-relationships/multi-table) for generating synthetic data, which uses the [mostlyai Python package](https://github.com/mostly-ai/mostly-python).

## Model Configuration

We use the following parameters when setting up the generators:

| **Parameter**                     | **Value**               |
|-----------------------------------|-------------------------|
| model                             | MOSTLY_AI/Large         |
| maxSampleSize                     | None                   |
| batchSize                         | None                   |
| maxTrainingTime                   | 120                    |
| maxEpochs                         | 100                    |
| maxSequenceWindow                 | 100                    |
| enableFlexibleGeneration          | False                  |
| valueProtection                   | False                  |
| rareCategoryReplacementMethod     | CONSTANT               |
| differentialPrivacy               | None                   |


## A Note on Reproducibility
> The platform does not provide an option to set a seed for reproducibility. Therefore, the generated data will vary from our runs. For this reason we provide scripts for generating the data generated in our runs in the `experiments/data` directory.
