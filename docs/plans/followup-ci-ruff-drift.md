# Follow-up: CI `validate` is red on unpinned-ruff drift

**Status:** the pin (decision 1) has landed; `TRY004` (decision 2) is still open.

`main`'s CI had been failing since at least 2026-07-19 — every run in that
window shows `validate` red while all four `test` matrix jobs pass. This is not
caused by any one branch.

## Root cause

`.github/workflows/ci.yml` installs `.[dev]`, and `pyproject.toml`'s `dev` extra
pins no ruff version:

```toml
dev = ["bandit", "build", "flit_core>=3.12,<4", "mypy", "pandas-stubs",
       "pytest", "ruff", "twine"]
```

CI therefore resolves whatever ruff is current at run time. Local installs lag
(0.15.5 as of this writing). Newer ruff enables rules the older one does not, so
`python -m ruff check .` passes locally and fails in CI on code nobody changed.
The failure set grows silently as ruff releases.

As observed on 2026-08-07, CI reported 9 errors, 8 of them in files untouched
for weeks:

| Rule | Location | Count |
|---|---|---|
| `TRY004` prefer `TypeError` for invalid type | `impact_split/splitter.py` (44, 98, 185, 191, 193) | 5 |
| `TRY004` | `impact_split/viz/data.py:49` | 1 |
| `TRY004` | `impact_split/viz/text.py` (47, 51) | 2 |

The 9th (`RUF046`, a redundant `int()` around `len()`) was introduced by the
relational-input branch and fixed there in `63c47d2`.

## Two decisions, not one

**1. Stop the drift. — Done.** The `dev` extra now pins `ruff~=0.15.5`, a
compatible-release pin so patch fixes still arrive while the minor version is
bumped deliberately. CI had resolved 0.16.1 against a local 0.15.5; under the
pin both report zero errors. Bumping the pin is now a change that carries its
own findings, rather than a surprise on an unrelated branch.

**2. Decide on `TRY004` separately.** The rule wants `TypeError` rather than
`ValueError` for wrong-type arguments. That is a **public API change** — callers
catching `ValueError` on `ImpactSplitter(...)` or `fit(...)` would break — so it
is not a lint cleanup and should not be done reflexively to satisfy a linter.
Either adopt it as an intentional breaking change with a CHANGELOG entry, or add
`TRY004` to a per-file ignore with a comment explaining the API-stability
reason.

Note the interaction: `impact_split/schema.py`'s `SchemaError` subclasses
`ValueError`, matching the existing convention. Whatever is decided here should
apply to that module too, so the package raises one consistent way.

## Related

See [followup-ruff-tests-scope.md](followup-ruff-tests-scope.md) — a separate
deferred lint item. Doing both in one PR would be reasonable; doing either
inside a feature branch would not.
