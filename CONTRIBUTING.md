# Contributing to SyntheRela

We welcome and appreciate contributions to SyntheRela from the community. Please reach out if you have something in mind. We expect our handling of contributions to become more streamlined as the project matures.

## Issues, Bug Fixes, and Discussions

Please submit GitHub issues and open Pull Requests if you find bugs or other issues in SyntheRela. When reporting bugs, please include:

- A clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Your environment details (Python version, OS, etc.)

## Development Setup

To contribute to SyntheRela, follow these steps:

1. Fork the repository on GitHub

2. Clone your fork locally:

   ```bash
   git clone https://github.com/yourusername/syntherela.git
   cd syntherela
   ```

3. Install the package in development mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

5. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Quality

We use several tools to maintain code quality:

- **pre-commit**: Automatically runs checks before each commit
- **ruff**: For linting and code formatting
- All code should follow PEP 8 style guidelines
- Add docstrings to new functions and classes
- Include type hints where appropriate

<!-- ## Contributing Synthetic Data Generation Methods

SyntheRela serves as a benchmark for synthetic relational data generation. We welcome contributions of new synthetic data generation methods. To add a new method:

1. Create a new file in the appropriate module following existing patterns
2. Implement your method following the established interface
3. Add comprehensive docstrings and comments
4. Include unit tests for your implementation
5. Update relevant documentation -->

## Contributing Evaluation Metrics

We welcome new evaluation metrics for assessing synthetic relational data quality. When contributing metrics:

1. Follow existing metric implementation patterns
2. Provide clear documentation on what the metric measures

<!-- 3. Include examples of usage
4. Add appropriate unit tests
5. Consider computational efficiency for large datasets -->

<!-- ## Contributing Datasets

If you have relational datasets that would be valuable for the benchmark:

1. Ensure you have proper permissions to share the data
2. Follow privacy and ethical guidelines
3. Provide clear documentation about the dataset
4. Include data preprocessing scripts if needed
5. Add dataset description and metadata -->

## Pull Request Process

1. Ensure your code passes all pre-commit checks

<!-- 2. Add or update tests as necessary -->

2. Update documentation if you're changing functionality
3. Write a clear description of your changes in the PR
4. Link any relevant issues

<!-- ## Testing

Before submitting a PR:

1. Run the test suite locally (if available)
2. Ensure your code works with the supported Python versions (3.8-3.13)
3. Test your changes with different operating systems if possible -->

## Documentation

When making changes:

- Update the README.md if your changes affect usage
- Add docstrings to new functions and classes
- [Optional] Update any relevant examples or tutorials
- [Optional] Consider adding notebook examples for new features

## Getting Help

If you need help with your contribution:

- Open a GitHub issue with the "question" label
- Reach out to the maintainers via email
- Check existing issues and discussions for similar questions

## Code of Conduct

We expect all contributors to follow professional and respectful communication
