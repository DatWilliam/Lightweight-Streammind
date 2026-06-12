import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
PY = sys.executable


def run(cmd: str) -> None:
    print(f"\n{'='*70}\n>>> {cmd}\n{'='*70}\n")

    if subprocess.call(cmd, shell=True) != 0:
        sys.exit(1)


DATASET = "soccernet"
WEIGHTS = f"checkpoints/epfe_{DATASET}.pt"


# ------------- Build Cache (2 fps) -------------
# Erzeugt *_2fps.npz neben den .mkv; alte 25-fps-Caches bleiben unangetastet.
# run(f"{PY} -m utils.build_cache {DATASET}")


# ------------- Training (Mamba only; EMA has no learnable params) -------------
run(f"{PY} -m utils.train {DATASET} --epochs 100 --patience 20 --lr 1e-4")


# ------------- Tune: Mamba x {fixed, th, full} -------------
run(f"{PY} -m eval.tune {DATASET} val --weights {WEIGHTS} --epfe mamba --gate fixed")
run(f"{PY} -m eval.tune {DATASET} val --weights {WEIGHTS} --epfe mamba --gate th")
run(f"{PY} -m eval.tune {DATASET} val --weights {WEIGHTS} --epfe mamba --gate full")


# ------------- Tune: EMA x {fixed, th, full} -------------
# run(f"{PY} -m eval.tune {DATASET} val --epfe ema --gate fixed")
# run(f"{PY} -m eval.tune {DATASET} val --epfe ema --gate th")
# run(f"{PY} -m eval.tune {DATASET} val --epfe ema --gate full")


# ------------- Test-Eval (after entering best params per combo into config.py) -------------
# run(f"{PY} -m eval.eval_cached {DATASET} test --weights {WEIGHTS} --epfe mamba --gate fixed")
# run(f"{PY} -m eval.eval_cached {DATASET} test --weights {WEIGHTS} --epfe mamba --gate th")
# run(f"{PY} -m eval.eval_cached {DATASET} test --weights {WEIGHTS} --epfe mamba --gate full")
# run(f"{PY} -m eval.eval_cached {DATASET} test --epfe ema --gate fixed")
# run(f"{PY} -m eval.eval_cached {DATASET} test --epfe ema --gate th")
# run(f"{PY} -m eval.eval_cached {DATASET} test --epfe ema --gate full")


print("\n" + "=" * 70 + "\nFERTIG.\n" + "=" * 70)
