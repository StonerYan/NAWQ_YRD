"""
Run official rev1 figure scripts.
Usage (from WQI0627_k/):  python figure/run_all.py
Prerequisite: python src/rev1/run_figure_inputs.py --tag rev1
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
scripts = [
    "fig3_performance.py",
    "fig4_scatter.py",
    "fig5_structure.py",
    "figS1_protocols.py",
    "figS4_oof.py",
    "fig7_conformal.py",
    "fig8_codmn_mech.py",
    "fig9_nh3n_mech.py",
    "fig10_tp_mech.py",
    "fig6_importance.py",
    "figS3_multilevel.py",
]

failed = []
for s in scripts:
    print(f"\n{'=' * 50}\nRunning {s} …")
    result = subprocess.run([sys.executable, str(HERE / s)])
    if result.returncode != 0:
        print(f"  [WARN] {s} exited with code {result.returncode}")
        failed.append(s)

print("\nAll figure scripts finished. Check figure/output/")
if failed:
    print("Failed:", ", ".join(failed))
    sys.exit(1)
