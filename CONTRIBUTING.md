# Contributing

Thank you for helping improve impact-split.

## Environment

- Python **3.13.x** (see `requires-python` in `pyproject.toml`, currently `~=3.13.0`).
- A virtual environment is recommended.

## Install for development

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra installs linting, tests, and packaging tools (`ruff`, `pytest`, `build`, `twine`, and related).

## Local checks (same shape as CI)

From the repository root, either run the full gate:

```bash
make ci-check
```

or the individual steps that [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs:

```bash
python -m ruff check .
python -m pytest
python -m build --no-isolation
python -m twine check dist/*
```

`make lint` / `make test` / `make package-check` run subsets of the above (see [`Makefile`](Makefile)).

## Documentation

MkDocs sources live under [`docs/`](docs/). See [`docs/README.md`](docs/README.md) for how to build and preview the site locally.

## Pull requests

- Open a PR against the default branch with a short description of the change and why.
- Ensure `make ci-check` (or the CI-equivalent commands) passes before requesting review.
- For larger behavior or API changes, open an issue first to align on direction.
