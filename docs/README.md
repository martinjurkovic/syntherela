# SyntheRela Documentation

This directory contains the Sphinx documentation for SyntheRela.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -e ".[docs]"
```

Or install Sphinx and dependencies directly:

```bash
pip install -r docs/requirements.txt
```

### Building HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to view it.

### Other Build Formats

Sphinx supports multiple output formats:

```bash
make latexpdf  # Build PDF documentation
make epub      # Build EPUB documentation
make man       # Build man pages
make help      # See all available formats
```

### Cleaning Build Artifacts

To remove all build artifacts:

```bash
make clean
```

## Documentation Structure

- `index.rst` - Main documentation entry point
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `guides/` - User guides (adding metrics, replicating results)
- `api/` - API reference documentation (auto-generated from docstrings)
- `conf.py` - Sphinx configuration
- `requirements.txt` - Documentation build dependencies

## ReadTheDocs

The documentation is configured to be automatically built and hosted on ReadTheDocs. The configuration is in `.readthedocs.yml` at the repository root.

## Contributing to Documentation

When adding new modules or functions:

1. Add NumPy-style docstrings to your code
2. The API documentation will be automatically generated
3. If needed, add new `.rst` files in the `api/` directory
4. Update `index.rst` to include new sections in the table of contents
