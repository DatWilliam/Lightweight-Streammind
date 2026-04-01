# python -m eval.eval_cached epickitchen test
# python -m eval.eval_cached soccernet test
# python -m eval.eval_cached soccernet test --weights checkpoints/epfe.pt
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import load_config
from model.epfe_cached import EPFECached
from model.gate import EventGate
from utils.eval_func import calculate_timval, calculate_triggeracc
from utils.build_cache import clip_model_suffix


def run_eval(dataset: str, split: str, weights: str = None, use_mamba: bool = True):
    if dataset == "epickitchen":
        from data.prepare_epickitchen import load_video_labels, get_total_gt_events, get_frame_count
    else:
        from data.prepare_soccernet import load_video_labels, get_total_gt_events, get_frame_count

    config = load_config(dataset)
    video_ids = getattr(config, f"video_ids_{split}")

    all_trigger_frames = []
    found_events = 0

    for video_id in video_ids:
        suffix = clip_model_suffix(config.clip_model)
        cache_path = config.DATA_DIR / dataset / str(Path(video_id).with_suffix(f".{suffix}.npz"))
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache nicht gefunden: {cache_path}\n"
                f"Erst ausführen: python -m utils.build_cache {dataset}"
            )

        gate = EventGate(config)
        epfe = EPFECached(config, use_mamba=use_mamba)
        if weights:
            epfe.load_weights(weights)

        gt_events = load_video_labels(video_id)
        trigger_frames = []

        data = np.load(str(cache_path))
        for frame_idx, feature in tqdm(
            zip(data["frame_idx"].tolist(), data["features"]),
            total=len(data["frame_idx"]),
            desc=video_id
        ):
            result = epfe.process_frame(feature)
            triggered = gate.check_event(result, int(frame_idx))
            if triggered is not False:
                trigger_frames.append(int(triggered))

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
    parser.add_argument("dataset", choices=["epickitchen", "soccernet"], default="soccernet", nargs="?")
    parser.add_argument("split", choices=["train", "test"], default="test", nargs="?")
    parser.add_argument("--weights",  default=None, help="Pfad zu trainierten Gewichten (.pt)")
    parser.add_argument("--no_mamba", action="store_true", help="Ablation: Mamba weglassen")
    args = parser.parse_args()
    run_eval(args.dataset, args.split, args.weights, use_mamba=not args.no_mamba)