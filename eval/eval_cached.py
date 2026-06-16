import argparse
import importlib
import numpy as np
from config import load_config
from utils.eval_func import per_video_metrics, macro_average, count_phase_fp

GATE_MODULES = {
    "full":  "model.gate", # confirmation gate
    "th":    "model.gate_th",
    "fixed": "model.gate_fixed",
}
EPFE_MODULES = {
    "mamba": ("model.epfe_mamba_cached", "EPFECached"),
    "ema":   ("model.epfe_ema_cached",   "EPFEEMACached"),
}


def run_eval(dataset: str, split: str, weights: str = None, use_mamba: bool = True,
             gate_mode: str = "full", epfe_mode: str = "mamba", alpha: float = None):
    if dataset == "ego4d":
        from data.prepare_ego4d import load_video_labels, get_cache_path
    else:
        from data.prepare_soccernet import load_video_labels, get_cache_path

    # dynamically load gate and EPFE module per flag
    EventGate = importlib.import_module(GATE_MODULES[gate_mode]).EventGate
    epfe_module, epfe_class = EPFE_MODULES[epfe_mode]
    EPFECls = getattr(importlib.import_module(epfe_module), epfe_class)
    if epfe_mode == "ema":
        print(f"EPFE: ema (alpha={alpha if alpha is not None else load_config(dataset).alpha}) | Gate: {gate_mode}")
    else:
        print(f"EPFE: {epfe_mode} | Gate: {gate_mode}")

    config = load_config(dataset)
    video_ids = getattr(config, f"video_ids_{split}")
    if alpha is not None:
        config.alpha = alpha

    epfe = EPFECls(config, use_mamba=use_mamba)
    epfe.eval()
    if weights:
        epfe.load_weights(weights)

    per_video = []
    skipped_no_gt = 0

    for video_id in video_ids:
        cache_path = get_cache_path(video_id)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run first: python -m utils.build_cache {dataset}"
            )

        gt_events = load_video_labels(video_id)
        data = np.load(str(cache_path))

        # EPFE scores for all cached frames in one go
        scores = epfe.score_video(data["features"])

        # send gate through scores frame-by-frame, collect triggers
        gate = EventGate(config)
        trigger_frames = []
        for frame_idx, score in zip(data["frame_idx"].tolist(), scores):
            result = {"event_score": float(score), "perception_token": None}
            triggered = gate.check_event(result, int(frame_idx))
            if triggered is not False:
                trigger_frames.append(int(triggered))

        # greedy 1:1-matching GT
        used_triggers = set()
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)

        # for metrics
        tp = len(used_triggers)
        false_triggers = [t for t in trigger_frames if t not in used_triggers]
        gt_starts = [e["start_frame"] for e in gt_events]
        fp_phase = count_phase_fp(false_triggers, gt_starts)
        n_frames = int(data["frame_idx"][-1])

        # Ignore videos with no GT event
        v = per_video_metrics(tp, fp_phase, len(gt_events), len(trigger_frames), n_frames)
        if v is None:
            skipped_no_gt += 1
            continue
        per_video.append(v)

    # macro-avg: store metrics per video
    avg = macro_average(per_video)
    metrics = {
        "F1_SCORE":         round(avg["f1"],          3),
        "recall":           round(avg["recall"],      3),
        "precision":        round(avg["precision"],   3),
        "trigger_acc":      round(avg["trigger_acc"], 3),
        "tim_val":          round(avg["tim_val"],     3),
        "call_red_percent": round(avg["call_red"] * 100, 2),
    }
    print(f"Macro-average ueber {len(per_video)} Videos"
          + (f" ({skipped_no_gt} ohne GT-Events uebersprungen)" if skipped_no_gt else ""))
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["soccernet", "ego4d"], default="soccernet", nargs="?")
    parser.add_argument("split", choices=["train", "val", "test"], default="test", nargs="?")
    parser.add_argument("--weights",  default=None, help="path to trained weights (.pt)")
    parser.add_argument("--no_mamba", action="store_true", help="ablation: drop Mamba")
    parser.add_argument("--gate", choices=list(GATE_MODULES.keys()), default="full",
                        help="gate mode: fixed | th (adaptive) | full (adaptive + confirm)")
    parser.add_argument("--epfe", choices=list(EPFE_MODULES.keys()), default="mamba",
                        help="EPFE backbone: mamba (trained) | ema (parameter-free)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="EMA decay (overrides config.alpha; only used by --epfe ema)")
    args = parser.parse_args()
    run_eval(args.dataset, args.split, args.weights,
             use_mamba=not args.no_mamba, gate_mode=args.gate, epfe_mode=args.epfe,
             alpha=args.alpha)
