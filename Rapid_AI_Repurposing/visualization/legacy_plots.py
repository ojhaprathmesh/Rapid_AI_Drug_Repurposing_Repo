"""
legacy_plots.py (DEPRECATED)
============================
This script has been merged into `generate_all_figures.py`.
All figure generation (including the 11-metric detailed evaluation breakdown,
formerly generated here) is now consolidated in:
    Rapid_AI_Repurposing.visualization.generate_all_figures

This file is maintained for backward compatibility.
"""

import sys
import os

# Ensure package root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from .generate_all_figures import gen_detailed_metrics_bar
except ImportError:
    from generate_all_figures import gen_detailed_metrics_bar

if __name__ == "__main__":
    print("[INFO] legacy_plots.py is merged into generate_all_figures.py. Calling gen_detailed_metrics_bar()...")
    gen_detailed_metrics_bar()

