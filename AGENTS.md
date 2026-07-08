# SyntheRela — Agent Instructions

## Project Overview

SyntheRela is a Python benchmark library for evaluating synthetic relational database generation methods. It provides fidelity and utility metrics across single-column, single-table, and multi-table granularities, enabling reproducible comparison of synthetic data generators on real-world datasets.

## Project Focus & Scope

- Prioritize changes in `syntherela/` (core library code) unless the prompt explicitly targets `experiments/`.
- Keep solutions minimal and scoped to the user request; prefer small, composable functions over broad rewrites.
- Preserve public APIs unless a change is explicitly requested.
- Do not refactor unrelated modules. Keep diffs concise, favouring root-cause fixes over workaround-only patches.

## Tech Stack

- **Language:** Python ≥ 3.10 (3.10–3.13 supported; 3.14+ not supported)
- **Core deps:** `sdmetrics ≥ 0.21`, `graphviz ≥ 0.13.2`, `scikit-learn > 1.3.1, < 1.5`, `xgboost == 1.7.6` (pinned), `seaborn == 0.13.2` (pinned)
- **Optional deps:** `sdv ≥ 1.9, < 2` (extra `[sdv]`, only for downloading SDV demo datasets)
- **Testing:** pytest, pytest-cov (≥ 85% coverage enforced)
- **Linting/formatting:** Ruff (lint + format), `ty` (type checker)
- **Docs:** Sphinx with NumPy-style docstrings
- **Build:** setuptools via `pyproject.toml` (no `setup.py`)

## Directory Structure

```
syntherela/          Main package
  benchmark.py       Core Benchmark class
  data.py            Dataset loading helpers
  metadata.py        Native Metadata / SingleTableMetadata (SDV-spec compatible)
  typing.py          Shared type aliases (Tables, etc.)
  metrics/           All metric implementations
    base.py          Abstract base classes — read before adding metrics
    single_column/   Column-level metrics (statistical/, distance/, detection/)
    single_table/    Table-level metrics
    multi_table/     Relational/cross-table metrics
  visualisations/    Plotting utilities (excluded from coverage)
tests/               pytest suite — mirrors syntherela/ structure
  conftest.py        Shared fixtures
  data/              Small fixture datasets
docs/                Sphinx source
experiments/         Research scripts (not part of the package)
examples/            Jupyter notebooks
```

## Install & Build

```bash
pip install -e ".[dev]"          # development (includes pytest, ruff, ty, pre-commit)
pip install -e ".[docs]"         # add Sphinx deps
pip install -e ".[rdl-utility]"  # add RDL benchmark deps
python -m build                  # produce dist/ wheel/sdist
```

## Commands

```bash
pytest                           # run all tests
pytest -m "not slow"             # skip slow tests
pytest --cov=syntherela          # with coverage (must stay ≥ 85%)
ruff check syntherela tests      # lint
ruff format syntherela tests     # auto-format
ty check --error-on-warning      # type check
pre-commit run --all-files       # run all hooks (format + lint + type + md)
cd docs && make html             # build Sphinx docs
```

## Code Conventions

- **Naming:** PascalCase classes, snake_case functions/methods, `_private` prefix for internal helpers.
- **Imports:** stdlib → third-party → local; use `isort` ordering (enforced by Ruff rule `I`).
- **Line length:** 80 characters (Ruff enforced; `experiments/` exempt from E501).
- **Quotes:** Single quotes (Ruff format setting).
- **Docstrings:** NumPy style on all public classes and methods. Module-level docstrings required.
- **Type hints:** Required on all function signatures; use Python 3.10+ syntax (`X | Y`, not `Optional[X]`).
- **Exports:** Every `__init__.py` must declare `__all__`.
- **Metric inheritance:** New metrics must inherit from the appropriate combination of base classes in `metrics/base.py` (e.g., `DistanceBaseMetric` + `SingleColumnMetric`). Reuse the existing abstractions (`SingleColumnMetric`, `SingleTableMetric`, `MultiTableMetric`, `StatisticalBaseMetric`, `DistanceBaseMetric`, `DetectionBaseMetric`) and follow the patterns in `docs/ADDING_A_METRIC.md`. Keep result keys/schemas consistent with existing metrics.

## Dependencies

- Use the standard library or existing project dependencies first.
- New dependencies are allowed only when they provide clear value and cannot reasonably be avoided; keep them lightweight and update packaging (`pyproject.toml`) consistently.
- Respect the pinned/ranged constraints listed under Hard Constraints.

## Workflow Requirements

- **After every code change:** run `pre-commit run --all-files` and fix any failures before considering the task done.
- **After changing a function or method signature, return type, parameters, or behaviour:** update its NumPy-style docstring to match — parameters, returns, raises sections must stay in sync with the implementation.
- **Always update documentation alongside code changes** (see Documentation Updates below) — docs are not a follow-up task.
- **Validation:** prefer targeted checks for the touched code paths (focused lint/format/type checks and the smallest relevant test) before broadening to full-repo runs. If a command is expensive (large data runs), propose it rather than running it automatically.

## Documentation Updates

Whenever behaviour or public usage changes, update the docs in the same change:

- `README.md` for user-facing usage changes.
- `docs/ADDING_A_METRIC.md` when metric extension patterns change.
- Relevant pages in `docs/` for reproducibility or setup changes.
- Keep the Sphinx docs (`docs/*.rst`, `docs/guides/*.rst`, `docs/api/*.rst`) in sync with `README.md` and the codebase:
  - When the README usage example changes, update `docs/index.rst` Quick Start and `docs/quickstart.rst` accordingly.
  - When new top-level README sections are added (e.g. Examples, Leaderboard), add matching pages under `docs/`.
  - When new public modules or classes are added, add a corresponding entry in the relevant `docs/api/*.rst` file.
- Build the docs locally (`cd docs && make html`) to verify there are no Sphinx errors after any documentation change.

## Hard Constraints

- **Do not** commit directly to `main` (pre-commit hook blocks it).
- **Do not** edit files under `syntherela.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `htmlcov/`, `docs/_build/` — all generated.
- **Do not** use `Optional[X]` or `Union[X, Y]` — use `X | None` and `X | Y` (Python 3.10+ style).
- **Do not** add metrics outside the `single_column/`, `single_table/`, `multi_table/` hierarchy without updating `metrics/base.py`.
- **Do not** reduce coverage below 85%; `visualisations/` is the only omitted directory.
- **Do not** break reproducibility scripts in `experiments/reproducibility/` — keep them stable and backward-compatible.
- **Do not** hardcode machine-specific paths; prefer project-relative paths and config.

## References

- [SyntheRela](https://openreview.net/forum?id=Mi8XioazWy)
- [Initial Benchmark](https://arxiv.org/abs/2410.03411)
- [SDV](https://docs.sdv.dev/sdv)
