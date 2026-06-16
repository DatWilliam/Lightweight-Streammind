# Lightweight StreamMind

A slim re-implementation of [StreamMind](https://github.com/xinding-sys/StreamMind), built for my bachelor thesis.
The pipeline decides per frame of a video stream whether a downstream LLM should be invoked,
so the LLM only runs when something semantically meaningful happens.

## Pipeline

```
frame  ->  CLIP (frozen)  ->  EPFE  ->  event score  ->  Gate  ->  trigger?
```

**EPFE** — two interchangeable variants:
- **Mamba** (`model/epfe_mamba_cached.py`): trainable Mamba block + linear score head on top of CLIP features.
- **EMA** (`model/epfe_ema_cached.py`): parameter-free L2 distance to an exponentially smoothed feature state.

**Gate** — three variants, all sharing a cooldown mechanism:
- **fixed** (`model/gate_fixed.py`): static threshold.
- **th** (`model/gate_th.py`): adaptive threshold `mean + k·std` over a sliding window.
- **full** (`model/gate.py`): adaptive threshold + two-stage confirmation.

## Datasets

Both datasets are subsampled to ~2 fps for comparability:
- **SoccerNet** (`25 fps -> 2.083 fps`, `sample_stride=12`)
- **Ego4D** (`30 fps -> 2 fps`, `sample_stride=15`)

The 70/15/15 train/val/test splits and all gate/EPFE hyperparameters live in `config.py`.

## Repository layout

```
config.py                      hyperparameters, sweep ranges, dataset splits
main.py                        thin runner with prewritten cache/train/tune/eval commands
data/
  prepare_ego4d.py             Ego4D path + narration parsing
  prepare_soccernet.py         SoccerNet path + Labels-v2.json parsing
  dataset.py                   PyTorch dataset for Mamba training
  download_soccernet.py        helper to fetch SoccerNet videos
model/
  epfe_mamba_cached.py         trainable Mamba EPFE (uses .npz cache)
  epfe_ema_cached.py           parameter-free EMA EPFE (uses .npz cache)
  gate*.py                     three gate variants
utils/
  build_cache.py               precompute CLIP features per video -> .npz
  train.py                     Mamba EPFE training (focal loss, early stopping)
  eval_func.py                 metrics (F1, recall, precision, call-red, TimVal, TriggerAcc)
eval/
  eval_cached.py               single-combo evaluation on the cached pipeline
  tune.py                      grid search over gate (+ alpha) on val, cross-check on test
live_test/
  epfe_mamba.py / epfe_ema.py  live EPFE variants (CLIP forward per frame)
  eval.py                      live evaluation, used for runtime measurements
```

## How to run

`main.py` is a thin wrapper with all relevant commands prewritten — comment lines in/out as needed, then:

```bash
python main.py
```

Typical workflow:
1. `utils.build_cache <dataset>` — precompute CLIP features (one-off, ~hours).
2. `utils.train <dataset>` — train Mamba EPFE (only needed for the Mamba variant).
3. `eval.tune <dataset> val --epfe <mamba|ema> --gate <fixed|th|full>` — grid-search gate hyperparameters.
4. After choosing the best params, run `eval.eval_cached <dataset> test ...` for the final number.

## Live (runtime) evaluation

`live_test/` contains the theoretically-correct pipeline that runs CLIP per incoming frame
(rather than from cache). It is used on the edge device (Jetson Orin NX, separate `orinnx` branch)
for runtime/throughput analysis.

## Acknowledgments

Based on [StreamMind](https://github.com/xinding-sys/StreamMind).
