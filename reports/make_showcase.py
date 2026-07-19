"""Regenerate the README showcase assets from one seeded synthetic fit.

Every artifact the README shows as "real output" is produced here, so the
README can never drift from what the package actually prints.

Why a bespoke demo instead of ``benchmarks.dgp``: that module's
``case_baseline`` is the robustness battery's regression guard — six
overlapping rules spread across three correlated dimensions (region, channel,
product), tuned to stress the tree, not to read cleanly as a first
impression. This demo plants exactly two independent, high-contrast rules
(one 2-way interaction, one main effect) at an effect size well above the
noise floor, so the top of the ledger is legible to a reader who has never
seen the library before.

Planted rules (ground truth, used only to make the demo legible — the fit
sees the outcome column alone):
  * region=West & channel=Online   -> +55 per row
  * segment=Enterprise             -> -35 per row
  * everything else                -> mean 0, noise sigma 18

Two DGP knobs were tuned away from the first draft after inspecting real
output (as Step 6 of the plan requires): effect sizes were raised from an
initial 40/-25 (with sigma 30) to 55/-35 (with sigma 18) because the weaker
draft buried the true drivers under sampling noise — 5 of 6 reported
segments came back flagged as noise churn, including the top one. A
``tenure_months`` numeric column was tried and dropped: it carries no
planted effect, so its only visible effect was a spurious micro-split
("tenure_months=12 (+50 more)" on 7 rows) that added clutter without adding
narrative value.

Usage:  python reports/make_showcase.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402  (must follow the backend selection)
import pandas as pd  # noqa: E402

from impact_split import ImpactSplitter  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent
SHOWCASE_DIR = REPORTS_DIR / "showcase"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
N_ROWS = 6_000
NOISE_SIGMA = 18.0
WEST_ONLINE_EFFECT = 55.0
ENTERPRISE_EFFECT = -35.0


def build_demo() -> tuple[pd.DataFrame, np.ndarray]:
    """A synthetic profit book with two planted drivers and no interaction between them.

    See the module docstring for the planted rules and why this demo does not
    reuse ``benchmarks.dgp.case_baseline``.
    """
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(
        {
            "region": rng.choice(["West", "East", "North", "South"], N_ROWS),
            "channel": rng.choice(["Online", "Retail", "Partner"], N_ROWS),
            "segment": rng.choice(["SMB", "Mid-Market", "Enterprise"], N_ROWS),
        }
    )
    y = rng.normal(0.0, NOISE_SIGMA, N_ROWS)
    y += np.where(
        (frame["region"] == "West") & (frame["channel"] == "Online"), WEST_ONLINE_EFFECT, 0.0
    )
    y += np.where(frame["segment"] == "Enterprise", ENTERPRISE_EFFECT, 0.0)
    return frame, y


def _fitted() -> ImpactSplitter:
    frame, y = build_demo()
    return ImpactSplitter().fit(frame, y)


def render_summary_text() -> str:
    return str(_fitted())


def _to_markdown(df: pd.DataFrame) -> str:
    """Hand-rolled markdown table (``tabulate`` is not a project dependency)."""
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.1f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def render_segments_markdown() -> str:
    segments = _fitted().get_impact_segments().head(8)
    return _to_markdown(segments)


def render_ensemble_text() -> str:
    frame, y = build_demo()
    model = ImpactSplitter().fit(frame, y)
    model.ensemble_report(frame, y, seed=SEED)
    return model.summary()


def main() -> None:
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    (SHOWCASE_DIR / "summary.txt").write_text(render_summary_text(), encoding="utf-8")
    (SHOWCASE_DIR / "segments.md").write_text(render_segments_markdown(), encoding="utf-8")
    (SHOWCASE_DIR / "summary-ensemble.txt").write_text(render_ensemble_text(), encoding="utf-8")

    model = _fitted()
    model.plot_segments(show=False).savefig(
        FIGURES_DIR / "segments-tornado.png", dpi=150, bbox_inches="tight"
    )
    model.plot_tree(show=False).savefig(
        FIGURES_DIR / "impact-icicle.png", dpi=150, bbox_inches="tight"
    )

    for name in ("summary.txt", "segments.md", "summary-ensemble.txt"):
        print(f"wrote {SHOWCASE_DIR / name}")
    for name in ("segments-tornado.png", "impact-icicle.png"):
        print(f"wrote {FIGURES_DIR / name}")


if __name__ == "__main__":
    main()
