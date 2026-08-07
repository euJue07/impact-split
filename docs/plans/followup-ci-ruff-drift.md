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
| `BLE001` do not catch blind `Exception` | `impact_split/viz/data.py:49` | 1 |
| `ISC004` unparenthesized implicit string concat in a collection | `impact_split/viz/text.py` (47, 51) | 2 |

*(Corrected: an earlier revision of this table attributed all 8 to `TRY004`.
The three `viz/` hits are different rules entirely, and the mistake made the
`TRY004` decision look ~60% larger and broader than it is. Re-read from the
run log rather than from this table if it matters.)*

The 9th (`RUF046`, a redundant `int()` around `len()`) was introduced by the
relational-input branch and fixed there in `63c47d2`.

## Two decisions, not one

**1. Stop the drift. — Done.** The `dev` extra now pins `ruff~=0.15.5` and
`mypy~=2.3.0`, compatible-release pins so patch fixes still arrive while the
minor version is bumped deliberately. CI had resolved ruff 0.16.1 against a
local 0.15.5; under the pin both report zero errors. Bumping is now a change
that carries its own findings, rather than a surprise on an unrelated branch.

Pinning was necessary but **not sufficient**, and the reason is worth keeping.
`validate` runs format → lint → mypy → bandit → build → twine and stops at the
first failure, so while ruff was red at step 2 **every later step had been dark
in CI since July**. Clearing ruff exposed two more, neither related to the pin:

- `ensemble.py` assigned `None` to `ci_low`/`ci_high`, inferred `float` from the
  percentile branch. Both were already documented as nullable and already
  reached the payload as `None`; only the annotation was missing.
- mypy was told `python_version = "3.10"` while the job ran on 3.13, where pip
  resolves numpy 2.5.1 — a release that itself requires ≥3.12 and whose stubs
  use PEP 695 `type` statements that mypy then refuses to parse as 3.10. That
  pairing cannot occur for a real user (a 3.10 install gets numpy ≤2.2.6), so
  the job now runs on 3.10 and the declared floor and the checked dependency set
  are the same thing.

The general lesson: a fail-fast job hides everything behind its first red step,
so the cost of a broken gate compounds. Treat the *first* green run after a long
red stretch as the real measurement, not the fix that produced it.

**2. Decide on `TRY004` separately. — Done: adopted.** The rule wants
`TypeError` rather than `ValueError` for wrong-type arguments. That is a
**public API change** — callers catching `ValueError` on `ImpactSplitter(...)`
or `fit(...)` break — so it was treated as a product decision, not a lint
cleanup.

Adopted as an intentional breaking change at the cheapest possible moment: 0.3.0
is still unreleased, so it ships inside a version nobody has pinned yet. Five
sites in `splitter.py`, four tests updated, CHANGELOG entry under
`### Changed (breaking)`. `TRY004` is now explicitly in `extend-select`, so the
convention is enforced under the pin instead of depending on ruff's default set.

`schema.py`'s `SchemaError` deliberately still subclasses `ValueError`: a schema
violation (duplicate key, unknown column, fan-out risk) is a wrong *value*, not
a wrong *type*, so the two modules do agree — the split is by fault kind, not by
module.

## Still open: `BLE001` and `ISC004`

The three `viz/` findings above are untouched and will fire again on a bump past
0.15.x. Both are smaller than `TRY004` and neither is an API change:

- `BLE001` at `viz/data.py:49` guards an `importlib.metadata` lookup that
  already carries a `# pragma: no cover` and deliberately degrades to
  `"unknown"`. A blind catch is arguably correct here; narrowing it or ignoring
  the rule are both defensible.
- `ISC004` at `viz/text.py` (47, 51) flags intentional line-wrapping of f-string
  headers inside a list literal. The rule exists because a missing comma in a
  collection silently becomes concatenation — worth a look before dismissing.

## Related

See [followup-ruff-tests-scope.md](followup-ruff-tests-scope.md) — a separate
deferred lint item. Doing both in one PR would be reasonable; doing either
inside a feature branch would not.
