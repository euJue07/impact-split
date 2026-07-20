"""Documentation link integrity.

Every relative markdown link and image target across the documentation set
must resolve to a path that exists. External (http/https/mailto) links and
bare anchors are out of scope — this guards against file moves and typos,
not link rot on the open web.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Matches [text](target) and ![alt](target).
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")


def _doc_files() -> list[Path]:
    """Every markdown file whose links we own."""
    files = [
        REPO_ROOT / name
        for name in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md")
    ]
    files += sorted((REPO_ROOT / "docs").glob("*.md"))
    files += sorted((REPO_ROOT / "docs" / "docs").glob("*.md"))
    files += sorted((REPO_ROOT / "reports").glob("*.md"))
    return [f for f in files if f.exists()]


def _relative_targets(doc: Path) -> list[str]:
    targets = []
    for match in LINK_RE.finditer(doc.read_text(encoding="utf-8")):
        # A markdown target may carry a title: [x](path "Title").
        target = match.group(1).split()[0].strip("<>")
        if target.startswith(SKIP_PREFIXES):
            continue
        targets.append(target)
    return targets


@pytest.mark.parametrize(
    "doc",
    _doc_files(),
    ids=lambda p: str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
)
def test_relative_links_resolve(doc: Path) -> None:
    broken = [
        target
        for target in _relative_targets(doc)
        if not (doc.parent / target.split("#")[0]).resolve().exists()
    ]
    assert not broken, f"{doc.relative_to(REPO_ROOT)} has broken relative links: {broken}"
