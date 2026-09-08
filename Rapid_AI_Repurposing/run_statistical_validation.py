#!/usr/bin/env python3
"""Root execution forwarder for pipeline/run_statistical_validation.py"""
import os, runpy
if __name__ == '__main__':
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline', 'run_statistical_validation.py')
    runpy.run_path(target, run_name='__main__')
