# Documentation (MkDocs)

This directory contains the **MkDocs** project for the documentation site. The
site is **not yet hosted** — build and preview it locally with the steps below.
Once deployed it will live at
[https://eujue07.github.io/impact-split/](https://eujue07.github.io/impact-split/).

Sources live in [`docs/docs/`](docs/) with configuration in [`mkdocs.yml`](mkdocs.yml).

## Prerequisites

- Python 3.10+ (same as the main package).
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

`mkdocs build` writes output to `site/` (ignored by git if listed in `.gitignore`).

**Do not use `mkdocs build --strict` here.** The documentation pages intentionally
cross-link to repository files that live *outside* `docs_dir` — the top-level
[`README.md`](../README.md), the `reports/` validation artifacts, `reports/figures/`
images embedded in the story, and the `notebooks/`. Those `../../…` links resolve
correctly in the GitHub file view, which is the primary reading surface while the
site is unhosted, but MkDocs cannot follow them within the built site, so `--strict`
aborts on ~13 out-of-tree-link warnings that are expected, not defects. Two further
site-only caveats to resolve if the site is ever hosted: the math on
[`math.md`](docs/math.md) uses `$…$` / ```` ```math ```` notation that renders on
GitHub but needs `pymdownx.arithmatex` + a MathJax `extra_javascript` entry to render
on the site; and the out-of-tree images/links would need copying into `docs/` or
rewriting. Until then, prefer the GitHub file view for these pages.

## Publishing to GitHub Pages

If this repository uses the standard GitHub Pages layout for MkDocs under `gh-pages`:

```bash
cd docs
mkdocs gh-deploy
```

Confirm branch and custom-domain settings in the GitHub repository **Settings → Pages** if your deployment path differs.
