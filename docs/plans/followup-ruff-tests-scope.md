# Follow-up: bring `tests/` into ruff's scope

Deferred out of the star/snowflake relational-input branch
(`claude/algorithm-relational-db-support-f9e2dd`). Cost already measured — this
is ready to execute as its own PR.

## Why it is not done yet

`[tool.ruff] include` in `pyproject.toml` is scoped to
`["pyproject.toml", "impact_split/**/*.py"]`, so `make lint` and CI's
`ruff check .` / `ruff format --check .` never reach `tests/`. That branch added
~800 lines of test code CI will not lint.

Adding `"tests/**/*.py"` was attempted on that branch and backed out. `include`
governs file discovery for **both** `ruff check` and `ruff format` — there is no
way to expose `tests/` to one and not the other — so the change surfaced
pre-existing drift in 8 files, 5 of them unrelated to the branch's work. A
mechanical sweep bundled with correctness changes is how real regressions get
waved through, so it was deferred rather than ridden along.

## What to do

1. Add `"tests/**/*.py"` back to `[tool.ruff] include` and remove the
   explanatory comment added in `6ecccd2`.
2. Run `ruff format tests/` over these 8 files:
   `test_benchmark_scoring.py`, `test_churn.py`, `test_ensemble.py`,
   `test_impact_splitter.py`, `test_viz_data.py`, `test_viz_html.py`,
   `test_viz_static.py`, `test_viz_text.py`.
3. Run the full suite after formatting. Five of those files have not been
   through a format pass before, so this is not a no-op verification.

## Nature of the drift

Whitespace and line-wrap only — multi-line calls (mostly kwarg-heavy ones like
`run_ensemble(...)` and `ceiling_cell(...)`) collapse or re-wrap to the
line-length-99 rule. No semantic changes expected.

`test_viz_html.py`, `test_viz_static.py`, and `test_viz_text.py` already had
their `ruff check` fixes landed on the relational-input branch (4 `I001` import
sorts and one `F841`); only the format pass is outstanding for those three.
