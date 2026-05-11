import argparse
import importlib
import numpy as np
from config import load_config
from utils.eval_func import calculate_timval, calculate_triggeracc

# gate_mode -> module name
GATE_MODULES = {
    "full":  "model.gate",
    "th":    "model.gate_th",
    "fixed": "model.gate_fixed",
}

# epfe_mode -> (module, class). EMA has no learnable params.
EPFE_MODULES = {
    "mamba": ("model.epfe_cached", "EPFECached"),
    "ema":   ("model.epfe_ema",    "EPFEEMACached"),
}


def run_eval(dataset: str, split: str, weights: str = None, use_mamba: bool = True,
             gate_mode: str = "full", epfe_mode: str = "mamba", alpha: float = None):
    if dataset == "ego4d":
        from data.prepare_ego4d import load_video_labels, get_total_gt_events, get_frame_count, get_cache_path
    else:
        from data.prepare_soccernet import load_video_labels, get_total_gt_events, get_frame_count, get_cache_path

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

    all_trigger_frames = []
    found_events = 0

    for video_id in video_ids:
        cache_path = get_cache_path(video_id)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache not found: {cache_path}\n"
                f"Run first: python -m utils.build_cache {dataset}"
            )

        gt_events = load_video_labels(video_id)
        data = np.load(str(cache_path))

        # bulk EPFE inference, then gate frame-by-frame in numpy
        scores = epfe.score_video(data["features"])

        gate = EventGate(config)
        trigger_frames = []
        for frame_idx, score in zip(data["frame_idx"].tolist(), scores):
            result = {"event_score": float(score), "perception_token": None}
            triggered = gate.check_event(result, int(frame_idx))
            if triggered is not False:
                trigger_frames.append(int(triggered))

        # match each GT to nearest unused trigger within tolerance
        used_triggers = set()
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)
                found_events += 1

        all_trigger_frames.extend(trigger_frames)

    llm_calls = len(all_trigger_frames)
    total_frames = get_frame_count(video_ids)
    call_red = 1 - (llm_calls / total_frames)
    total_gt = get_total_gt_events(video_ids)
    recall = found_events / total_gt
    precision = found_events / llm_calls if llm_calls > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

    metrics = {
        "F1_SCORE": round(f1, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "trigger_acc": max(0.0, calculate_triggeracc(llm_calls, found_events, total_gt)),
        "tim_val": max(0.0, calculate_timval(llm_calls, found_events, total_gt)["timval"]),
        "call_red_percent": round(call_red * 100, 2),
    }

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