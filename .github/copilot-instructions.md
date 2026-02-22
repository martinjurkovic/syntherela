# Copilot Instructions for SyntheRela

## Project focus
- Prioritize changes in `syntherela/` (core library code) unless the prompt explicitly targets `experiments/`.
- Keep solutions minimal and scoped to the user request.
- Preserve public APIs unless a change is explicitly requested.

## Coding conventions
- Follow existing style in this repository (PEP 8, Ruff-compatible formatting, descriptive names).
- Add or update docstrings for new public classes/functions.
- Use type hints for new or changed function signatures where practical.
- Prefer small, composable functions over broad rewrites.

## Metrics and benchmark architecture
- Reuse existing metric abstractions in `syntherela.metrics.base`:
  - `SingleColumnMetric`, `SingleTableMetric`, `MultiTableMetric`
  - `StatisticalBaseMetric`, `DistanceBaseMetric`, `DetectionBaseMetric`
- When adding metrics, follow patterns documented in `docs/ADDING_A_METRIC.md`.
- Keep outputs and keys consistent with existing metric result schemas.

## Dependencies
- Use standard library or existing project dependencies first.
- New dependencies are allowed only when they provide clear value and cannot be reasonably avoided.
- If adding a dependency, keep it lightweight and update packaging files consistently.

## Validation workflow
- Prefer targeted validation for touched code paths instead of always running full-repo checks.
- Typical checks:
  - Run focused lint/format/type checks for changed files.
  - Run the smallest relevant script/test first, then broaden only if needed.
- If a command is expensive (large data runs), propose it instead of running automatically.

## Data and reproducibility constraints
- Do not modify files in `data/original/` or bulky artifacts unless explicitly requested.
- Keep reproducibility scripts in `experiments/reproducibility/` stable and backward-compatible.
- Avoid hardcoding machine-specific paths; prefer project-relative paths and config.

## Documentation updates
- Update docs when behavior or public usage changes:
  - `README.md` for user-facing usage changes.
  - `docs/ADDING_A_METRIC.md` when metric extension patterns change.
  - Relevant docs in `docs/` for reproducibility or setup changes.

## Safety and edit boundaries
- Do not refactor unrelated modules.
- Do not edit virtual environment contents (for example `.venv/`).
- Keep diffs concise, with root-cause fixes over workaround-only patches.