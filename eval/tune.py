import argparse
import importlib
import itertools
import numpy as np
from config import load_config
from utils.eval_func import per_video_metrics, macro_average, count_phase_fp

EPFE_MODULES = {
    "mamba": ("model.epfe_cached", "EPFECached"),
    "ema":   ("model.epfe_ema",    "EPFEEMACached"),
}


def _precompute_scores(config, dataset: str, video_ids: list, weights: str,
                        use_mamba: bool, epfe_mode: str = "mamba") -> dict:
    # load EPFE once, run bulk inference per video
    if dataset == "ego4d":
        from data.prepare_ego4d import load_video_labels, get_cache_path
    else:
        from data.prepare_soccernet import load_video_labels, get_cache_path

    epfe_module, epfe_class = EPFE_MODULES[epfe_mode]
    EPFECls = getattr(importlib.import_module(epfe_module), epfe_class)
    epfe = EPFECls(config, use_mamba=use_mamba)
    epfe.eval()
    if weights:
        epfe.load_weights(weights)

    video_scores = {}
    for video_id in video_ids:
        cache_path = get_cache_path(video_id)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run first: python -m utils.build_cache {dataset}"
            )
        data = np.load(str(cache_path))
        scores = epfe.score_video(data["features"])

        video_scores[video_id] = {
            "frame_idx": data["frame_idx"].tolist(),
            "scores": scores,
            "gt_events": load_video_labels(video_id),
            "total_frames": int(data["frame_idx"][-1]),
        }
    return video_scores


def _sliding_mean_std(scores: np.ndarray, window_size: int):
    # returns (win_mean, win_std) for each full window starting at index i
    shape = (len(scores) - window_size + 1, window_size)
    strides = (scores.strides[0], scores.strides[0])
    windows = np.lib.stride_tricks.as_strided(scores, shape=shape, strides=strides)
    return windows.mean(axis=1), windows.std(axis=1) + 1e-6


def _triggers_fixed(scores: np.ndarray, frame_idx: list, threshold: float, cooldown: int):
    # gate_fixed: score > threshold + cooldown
    trigger_frames = []
    last_event_frame = -cooldown
    for i, score in enumerate(scores):
        fidx = frame_idx[i]
        if score > threshold and fidx - last_event_frame >= cooldown:
            last_event_frame = fidx
            trigger_frames.append(fidx)
    return trigger_frames


def _triggers_th(scores: np.ndarray, frame_idx: list, k: float, window_size: int, cooldown: int):
    # gate_th: adaptive threshold (mean + k*std), no confirmation stage
    N = len(scores)
    if N < window_size:
        return []
    win_mean, win_std = _sliding_mean_std(scores, window_size)
    threshold = win_mean + k * win_std

    trigger_frames = []
    last_event_frame = -cooldown
    for i in range(window_size - 1, N):
        wi = i - (window_size - 1)
        score = scores[i]
        thr = threshold[wi]
        fidx = frame_idx[i]
        if score > thr and fidx - last_event_frame >= cooldown:
            last_event_frame = fidx
            trigger_frames.append(fidx)
    return trigger_frames


def _triggers_full(scores: np.ndarray, frame_idx: list,
                   k, window_size, confirm_frames, cooldown) -> list:
    # gate.py (full): adaptive threshold + 2-stage confirmation
    N = len(scores)
    if N < window_size:
        return []
    win_mean, win_std = _sliding_mean_std(scores, window_size)
    threshold = win_mean + k * win_std

    trigger_frames = []
    candidate_frame = None
    mean_at_spike = 0.0
    confirm_count = 0
    last_event_frame = -cooldown

    for i in range(window_size - 1, N):
        wi = i - (window_size - 1)
        score = scores[i]
        mean = win_mean[wi]
        thr = threshold[wi]
        fidx = frame_idx[i]

        if candidate_frame is not None:
            if score > mean_at_spike:
                confirm_count += 1
                if confirm_count >= confirm_frames:
                    last_event_frame = candidate_frame
                    trigger_frames.append(candidate_frame)
                    candidate_frame = None
                    confirm_count = 0
            else:
                candidate_frame = None
                confirm_count = 0

        if (score > thr and candidate_frame is None
                and fidx - last_event_frame >= cooldown):
            candidate_frame = fidx
            mean_at_spike = mean
            confirm_count = 0

    return trigger_frames


def _eval_gate(video_scores: dict, config, gate_mode: str = "full") -> dict:
    # run gate on precomputed scores, macro-average metrics across all videos
    per_video = []

    for v in video_scores.values():
        if gate_mode == "fixed":
            trigger_frames = _triggers_fixed(
                v["scores"], v["frame_idx"], config.fixed_threshold, config.cooldown,
            )
        elif gate_mode == "th":
            trigger_frames = _triggers_th(
                v["scores"], v["frame_idx"], config.k, config.window_size, config.cooldown,
            )
        else:  # full
            trigger_frames = _triggers_full(
                v["scores"], v["frame_idx"],
                config.k, config.window_size, config.confirm_frames, config.cooldown,
            )
        gt_events = v["gt_events"]
        used_triggers = set()
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers
                       and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)

        tp = len(used_triggers)
        false_triggers = [t for t in trigger_frames if t not in used_triggers]
        gt_starts = [e["start_frame"] for e in gt_events]
        fp_phase = count_phase_fp(false_triggers, gt_starts)

        m = per_video_metrics(tp, fp_phase, len(gt_events),
                              len(trigger_frames), v["total_frames"])
        if m is not None:
            per_video.append(m)

    avg = macro_average(per_video)
    return {
        "f1":          round(avg["f1"],          4),
        "recall":      round(avg["recall"],      4),
        "precision":   round(avg["precision"],   4),
        "call_red":    round(avg["call_red"],    4),
        "trigger_acc": round(avg["trigger_acc"], 4),
        "tim_val":     round(avg["tim_val"],     4),
    }


def _param_keys(gate_mode: str, epfe_mode: str = "mamba"):
    # which params show up in the results table
    cols = []
    if epfe_mode == "ema":
        cols.append(("alpha", "alpha", 8))
    if gate_mode == "fixed":
        cols.append(("threshold", "fixed_threshold", 10))
    elif gate_mode == "th":
        cols.extend([("k", "k", 7), ("ws", "window_size", 5)])
    else:  # full
        cols.extend([("k", "k", 7), ("cf", "confirm_frames", 5), ("ws", "window_size", 5)])
    return cols


def _print_results(results: list, top_n: int, gate_mode: str = "full", epfe_mode: str = "mamba"):
    pcols = _param_keys(gate_mode, epfe_mode)
    header = f"{'Rank':<6}" + "".join(f"{label:<{w}}" for label, _, w in pcols)
    header += f"{'F1':<8}{'recall':<9}{'prec':<8}{'call_red':<11}{'trig_acc':<11}{'tim_val'}"
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results[:top_n], 1):
        row = f"{rank:<6}" + "".join(f"{r[k]:<{w}}" for _, k, w in pcols)
        row += (f"{r['f1']:<8}{r['recall']:<9}{r['precision']:<8}"
                f"{r['call_red']:<11}{r['trigger_acc']:<11}{r['tim_val']}")
        print(row)


def _build_combos(config, gate_mode: str, k_values, confirm_frames_values, threshold_values,
                  window_size_values=None):
    # Default-Sweep-Bereiche aus der Dataset-Config; CLI-Overrides haben Vorrang.
    if gate_mode == "fixed":
        if threshold_values is None:
            threshold_values = config.fixed_threshold_sweep
        # fixed-Gate hat keine Sliding-Window-Statistik → window_size irrelevant.
        return [{"fixed_threshold": t} for t in threshold_values]
    if window_size_values is None:
        window_size_values = config.window_size_sweep
    if gate_mode == "th":
        if k_values is None:
            k_values = config.k_sweep
        return [{"k": k, "window_size": ws}
                for k, ws in itertools.product(k_values, window_size_values)]
    # full
    if k_values is None:
        k_values = config.k_sweep
    if confirm_frames_values is None:
        confirm_frames_values = config.confirm_frames_sweep
    return [{"k": k, "confirm_frames": cf, "window_size": ws}
            for k, cf, ws in itertools.product(k_values, confirm_frames_values, window_size_values)]


def tune_params(
    dataset: str,
    split: str,
    weights: str = None,
    use_mamba: bool = True,
    gate_mode: str = "full",
    epfe_mode: str = "mamba",
    alpha_values: list = None,
    k_values: list = None,
    confirm_frames_values: list = None,
    threshold_values: list = None,
):
    config = load_config(dataset)
    video_ids = getattr(config, f"video_ids_{split}")

    # alpha is an outer loop only for EMA (changes the scores)
    if epfe_mode == "ema":
        if alpha_values is None:
            alpha_values = config.alpha_sweep
    else:
        alpha_values = [None]

    combos = _build_combos(config, gate_mode, k_values, confirm_frames_values, threshold_values)
    total_combos = len(combos) * len(alpha_values)
    print(f"EPFE: {epfe_mode} | Gate: {gate_mode}")
    print(f"Sweep: {total_combos} combinations "
          f"({len(alpha_values)} alpha x {len(combos)} gate)")

    other_split = "test" if split in ("train", "val") else "train"
    other_ids = getattr(config, f"video_ids_{other_split}", [])

    # cache scores per alpha (or one entry for non-EMA)
    val_scores_by_alpha = {}
    test_scores_by_alpha = {}

    results = []
    progress = 0
    for ai, alpha_val in enumerate(alpha_values, 1):
        if alpha_val is not None:
            config.alpha = alpha_val
        video_scores = _precompute_scores(config, dataset, video_ids, weights, use_mamba, epfe_mode)
        val_scores_by_alpha[alpha_val] = video_scores

        for combo in combos:
            for key, val in combo.items():
                setattr(config, key, val)
            metrics = _eval_gate(video_scores, config, gate_mode)
            entry = {**combo, **metrics}
            if alpha_val is not None:
                entry["alpha"] = alpha_val
            results.append(entry)
            progress += 1

        # Heartbeat: ein Print pro Alpha-Durchlauf
        best = max(results, key=lambda x: x["f1"])
        tag = f"alpha={alpha_val} " if alpha_val is not None else ""
        print(f"  [{ai}/{len(alpha_values)}] {tag}-> {progress}/{total_combos} combos | best F1 so far: {best['f1']:.4f}")

    by_f1 = sorted(results, key=lambda x: x["f1"], reverse=True)
    by_tim = sorted(results, key=lambda x: x["tim_val"], reverse=True)
    top3_f1 = by_f1[:3]
    top3_tim = by_tim[:3]

    if not other_ids:
        return by_f1[0]

    # cross-check on held-out split: precompute test scores per used alpha
    used_alphas = {c.get("alpha") for c in top3_f1 + top3_tim}
    for alpha_val in used_alphas:
        if alpha_val is not None:
            config.alpha = alpha_val
        test_scores_by_alpha[alpha_val] = _precompute_scores(
            config, dataset, other_ids, weights, use_mamba, epfe_mode,
        )

    def _eval_candidates(candidates, label):
        param_keys = [k for _, k, _ in _param_keys(gate_mode, epfe_mode)]
        out = []
        for c in candidates:
            combo = {k: c[k] for k in param_keys}
            for key, val in combo.items():
                setattr(config, key, val)
            scores = test_scores_by_alpha[c.get("alpha")]
            metrics = _eval_gate(scores, config, gate_mode)
            out.append({**combo, **metrics})
        print(f"\n{label} on {other_split.upper()} (same order as {split.upper()}):")
        _print_results(out, len(out), gate_mode, epfe_mode)

    _eval_candidates(top3_f1, "Top 3 F1")
    _eval_candidates(top3_tim, "Top 3 TimVal")

    return by_f1[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",    choices=["soccernet", "ego4d"])
    parser.add_argument("split",      choices=["train", "val", "test"])
    parser.add_argument("--weights",  default=None, help="path to trained weights (.pt)")
    parser.add_argument("--no_mamba", action="store_true", help="ablation: drop Mamba")
    parser.add_argument("--gate",     choices=["full", "th", "fixed"], default="full",
                        help="gate mode: fixed | th (adaptive) | full (adaptive + confirm)")
    parser.add_argument("--epfe",     choices=list(EPFE_MODULES.keys()), default="mamba",
                        help="EPFE backbone: mamba (trained) | ema (parameter-free)")
    parser.add_argument("--alpha",    type=float, default=None,
                        help="EMA decay; if set, used as the single alpha (otherwise sweep)")
    args = parser.parse_args()

    alpha_values = [args.alpha] if args.alpha is not None else None
    tune_params(args.dataset, args.split, args.weights,
                use_mamba=not args.no_mamba,
                gate_mode=args.gate, epfe_mode=args.epfe,
                alpha_values=alpha_values)
