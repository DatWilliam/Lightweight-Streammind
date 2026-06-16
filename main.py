import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
PY = sys.executable


def run(cmd: str) -> None:
    print(f"\n{'='*70}\n>>> {cmd}\n{'='*70}")
    if subprocess.call(cmd, shell=True) != 0:
        sys.exit(1)


DATASET = "soccernet"

# ------------- Performance benchmark (batch=1 single-frame streaming, whole split) -------------
# throughput (fps), per-frame latency by component, peak RAM/VRAM
run(f"{PY} -m utils.bench {DATASET} --split test --warmup 50 --gate full")


# ------------- Live streaming eval (EMA, no cache): confirmation gate on test -------------
# run(f"{PY} -m eval.eval {DATASET} test --gate full")


print("\n" + "=" * 70 + "\nFERTIG.\n" + "=" * 70)
