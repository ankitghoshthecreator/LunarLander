"""Showcase frontend redirect to main dashboard."""

import runpy
from pathlib import Path

dashboard_path = Path(__file__).parent / "dashboard.py"
runpy.run_path(str(dashboard_path), run_name="__main__")
