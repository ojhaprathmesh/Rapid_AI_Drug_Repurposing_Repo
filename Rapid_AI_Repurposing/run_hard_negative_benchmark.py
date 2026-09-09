"""Root-level forwarder for run_hard_negative_benchmark.py"""
import runpy, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "pipeline", "run_hard_negative_benchmark.py"),
    run_name="__main__"
)
