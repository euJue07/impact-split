# Documentation (MkDocs)

This directory contains the **MkDocs** project for the published site:

[https://juedimyroeugenio.github.io/impact-split/](https://juedimyroeugenio.github.io/impact-split/)

Sources live in [`docs/docs/`](docs/docs/) with configuration in [`mkdocs.yml`](mkdocs.yml).

## Prerequisites

- Python 3.13+ (same as the main package).
- [MkDocs](https://www.mkdocs.org/) installed in the environment you use for docs work:

```bash
python -m pip install mkdocs
```

Optional: install from the repo root with dev extras (`python -m pip install -e ".[dev]"`) for the library, then add MkDocs as above if you only need the docs toolchain.

## Build and preview

Run all commands **from this directory** (`docs/`), where `mkdocs.yml` lives:

```bash
cd docs
mkdocs serve    # local preview, default http://127.0.0.1:8000/
mkdocs build    # static site under docs/site/
```

`mkdocs build` writes output to `site/` (ignored by git if listed in `.gitignore`). Use `mkdocs build --strict` in automation to fail on warnings (broken links, etc.).

## Publishing to GitHub Pages

If this repository uses the standard GitHub Pages layout for MkDocs under `gh-pages`:

```bash
cd docs
mkdocs gh-deploy
```

Confirm branch and custom-domain settings in the GitHub repository **Settings → Pages** if your deployment path differs.
