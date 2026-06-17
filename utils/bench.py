# Single-frame (batch=1) streaming benchmark for the EMA pipeline.

# Runs the whole split frame-by-frame and reports throughput (fps), per-frame
# latency split by component (decode | preprocess | encode | score | gate), and
# peak RAM/VRAM. GPU work is timed with cuda.synchronize() (CUDA is async), and
# the first `warmup` frames are excluded.

# Note: on Jetson the GPU shares the RAM pool (unified memory), so "VRAM" is just
# the CUDA allocation, not separate physical memory.
import argparse
import importlib
import resource
import time

import cv2
import numpy as np
import torch
from PIL import Image

from config import load_config
from model.epfe_ema import EPFE

GATE_MODULES = {
    "full":  "model.gate",
    "th":    "model.gate_th",
    "fixed": "model.gate_fixed",
}


# path lookup + frame-decode generator for the dataset
def _loaders(dataset: str):
    if dataset == "ego4d":
        from data.prepare_ego4d import get_video_path, load_video
    else:
        from data.prepare_soccernet import get_video_path, load_video
    return get_video_path, load_video


# list of per-frame seconds -> mean / median / p95 in ms
def _stats_ms(times_s):
    a = np.asarray(times_s, dtype=np.float64) * 1000.0  # -> ms
    return {
        "mean": float(a.mean()) if len(a) else 0.0,
        "p50":  float(np.percentile(a, 50)) if len(a) else 0.0,
        "p95":  float(np.percentile(a, 95)) if len(a) else 0.0,
    }


def run_bench(dataset: str, split: str, warmup: int, gate_mode: str, alpha: float = None):
    config = load_config(dataset)
    if alpha is not None:
        config.alpha = alpha

    video_ids = getattr(config, f"video_ids_{split}")
    get_video_path, load_video = _loaders(dataset)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    EventGate = importlib.import_module(GATE_MODULES[gate_mode]).EventGate

    print(f"Device: {device} | CLIP: {config.clip_model} | gate: {gate_mode} "
          f"| alpha: {config.alpha}")
    print(f"Split: {split} | videos: {len(video_ids)} | warmup: {warmup} frames | batch=1\n")

    epfe = EPFE(config)          # loads + freezes CLIP on the GPU
    model, preprocess = epfe.model, epfe.preprocess
    comps = {k: [] for k in ("decode", "preprocess", "encode", "score", "gate")}

    # one frame through preprocess -> encode -> score -> gate; record per-component times
    def step(frame, frame_idx, gate, state, record):
        # preprocess (CPU): BGR->RGB, PIL, CLIP transform
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0)
        t1 = time.perf_counter()

        # encode (GPU incl. host<->device transfer)
        with torch.no_grad():
            x = tensor.to(device)
            feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            if device == "cuda":
                torch.cuda.synchronize()
            feat = feat.squeeze(0).cpu().numpy()
        t2 = time.perf_counter()

        # score (EMA): L2 distance + state update
        if state is None:
            state = feat.copy()
            score = 0.0
        else:
            score = float(np.linalg.norm(feat - state))
            state = config.alpha * feat + (1.0 - config.alpha) * state
        t3 = time.perf_counter()

        # gate
        gate.check_event({"event_score": score, "perception_token": None}, frame_idx)
        t4 = time.perf_counter()

        if record:
            comps["preprocess"].append(t1 - t0)
            comps["encode"].append(t2 - t1)
            comps["score"].append(t3 - t2)
            comps["gate"].append(t4 - t3)
        return state

    processed = 0       # total frames seen (incl. warmup)
    measured = 0        # frames counted into stats
    wall0 = None
    n_videos = len(video_ids)

    for vi, video_id in enumerate(video_ids, 1):
        if not get_video_path(video_id).exists():
            print(f"[{vi}/{n_videos}] [skip] missing video: {video_id}", flush=True)
            continue

        # fresh gate + EMA state per video (matches eval.eval)
        gate = EventGate(config)
        state = None
        frame_idx = 0
        vid_measured = 0
        gen = load_video(video_id)

        # decode each frame here (timed), the rest happens in step()
        while True:
            td0 = time.perf_counter()
            try:
                frame = next(gen)
            except StopIteration:
                break
            td1 = time.perf_counter()
            frame_idx += 1

            # only count frames after warmup; start clock + mem peak on the first counted frame
            record = processed >= warmup
            if record and wall0 is None:
                # first measured frame: reset peak mem + start the wall clock
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                wall0 = time.perf_counter()
            if record:
                comps["decode"].append(td1 - td0)

            state = step(frame, frame_idx, gate, state, record)
            processed += 1
            if record:
                measured += 1
                vid_measured += 1

        name = get_video_path(video_id).name
        print(f"[{vi}/{n_videos}] {name}: {vid_measured} frames", flush=True)

    if wall0 is None or measured == 0:
        print("\nNo frames measured (warmup >= total frames?).")
        return
    wall = time.perf_counter() - wall0

    # ---- memory ----
    ram_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB->MB
    vram_alloc_mb = vram_reserved_mb = None
    if device == "cuda":
        vram_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
        vram_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2

    # ---- report ----
    per_frame_total_ms = (wall / measured) * 1000.0
    fps = measured / wall

    print("\n" + "=" * 60)
    print(f"Frames measured : {measured}  (over {n_videos} videos)")
    print(f"Wall time       : {wall:.1f} s")
    print(f"Throughput      : {fps:.1f} fps")
    print(f"Latency/frame   : {per_frame_total_ms:.2f} ms  (end-to-end, wall/frames)")
    print("-" * 60)
    print(f"{'component':<12}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}{'% total':>10}")
    comp_mean_sum = sum(_stats_ms(comps[c])["mean"] for c in comps)
    for c in ("decode", "preprocess", "encode", "score", "gate"):
        s = _stats_ms(comps[c])
        pct = 100.0 * s["mean"] / comp_mean_sum if comp_mean_sum else 0.0
        print(f"{c:<12}{s['mean']:>10.3f}{s['p50']:>10.3f}{s['p95']:>10.3f}{pct:>9.1f}%")
    print(f"{'sum':<12}{comp_mean_sum:>10.3f}")
    print("-" * 60)
    print(f"Peak RAM (RSS)        : {ram_peak_mb:.0f} MB")
    if device == "cuda":
        print(f"Peak VRAM (allocated) : {vram_alloc_mb:.0f} MB")
        print(f"Peak VRAM (reserved)  : {vram_reserved_mb:.0f} MB")
        print("(Jetson: unified memory — VRAM shares the physical RAM pool)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["soccernet", "ego4d"], default="soccernet", nargs="?")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--warmup", type=int, default=50, help="warmup frames (excluded)")
    parser.add_argument("--gate", choices=list(GATE_MODULES.keys()), default="full")
    parser.add_argument("--alpha", type=float, default=None, help="EMA decay (overrides config)")
    args = parser.parse_args()
    run_bench(args.dataset, args.split, args.warmup, args.gate, args.alpha)
