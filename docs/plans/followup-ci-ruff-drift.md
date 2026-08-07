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

## `ISC004` — done

Both hits in `viz/text.py` were checked before being silenced, which is the only
correct way to treat this rule: it fires on implicit concatenation inside a
collection precisely because a **missing comma looks identical to intentional
concatenation**, and one of the two readings is a silently wrong list. Here both
were intentional — one builds `"segments N · conservation X"`, the other a table
header row — so the fix is explicit parentheses, not a comma.

Evidence the change is invisible: `python -m reports.make_showcase` regenerates
`reports/showcase/*.txt` byte-identically (sha256 `ef8523fc4153a677` and
`b9be5d7f7fdde9db`, unchanged), and `git status reports/` stays clean.

Not added to `extend-select`: `ISC004` is a **preview** rule in 0.15.5 (ruff
prints "has no effect because preview is not enabled") and only stabilises in
0.16. Enabling `preview = true` to reach it would pull in 12 unrelated findings.
**Add `"ISC004"` to `extend-select` as part of the 0.16 bump**, where it is
stable and free.

## Still open: `BLE001`

- `BLE001` at `viz/data.py:49` guards an `importlib.metadata` lookup that
  already carries a `# pragma: no cover` and deliberately degrades to
  `"unknown"`. A blind catch is arguably correct here; narrowing it or ignoring
  the rule are both defensible. It will fire again on a bump past 0.15.x.
## Related

See [followup-ruff-tests-scope.md](followup-ruff-tests-scope.md) — done; that
sweep landed once the pin made a local `ruff format` and CI's
`ruff format --check` the same operation.
