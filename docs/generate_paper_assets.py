#!/usr/bin/env python3
"""
generate_paper_assets.py (Forwarder)
Delegates to Rapid_AI_Repurposing/visualization/generate_all_figures.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

from Rapid_AI_Repurposing.visualization.generate_all_figures import main

if __name__ == "__main__":
    main()
