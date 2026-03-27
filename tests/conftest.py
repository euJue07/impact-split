"""Pytest configuration: non-interactive matplotlib backend before any test imports."""

import matplotlib

matplotlib.use("Agg")
