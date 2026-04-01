# python -m eval.tune soccernet test --weights checkpoints/epfe_soccernet.pt
import argparse
import itertools
import numpy as np
from pathlib import Path
from config import load_config
from model.epfe_cached import EPFECached
from model.gate import EventGate


def _precompute_scores(config, dataset: str, video_ids: list, weights: str, use_mamba: bool) -> dict:
    """Lädt CLIP-Cache und berechnet Event-Scores einmalig für alle Videos."""
    if dataset == "epickitchen":
        from data.prepare_epickitchen import load_video_labels
    else:
        from data.prepare_soccernet import load_video_labels

    epfe = EPFECached(config, use_mamba=use_mamba)
    if weights:
        epfe.load_weights(weights)

    video_scores = {}

    for video_id in video_ids:
        cache_path = config.DATA_DIR / dataset / str(Path(video_id).with_suffix(".npz"))
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache nicht gefunden: {cache_path}\n"
                f"Erst ausführen: python -m utils.build_cache {dataset}"
            )

        epfe.reset()
        data = np.load(str(cache_path))
        scores = []
        for frame_idx, feature in zip(data["frame_idx"].tolist(), data["features"]):
            result = epfe.process_frame(feature)
            scores.append((int(frame_idx), result["event_score"]))

        video_scores[video_id] = {
            "scores": scores,
            "gt_events": load_video_labels(video_id),
            "total_frames": int(data["frame_idx"][-1]),
        }

    return video_scores


def _eval_gate(config, video_scores: dict) -> dict:
    """Evaluiert Gate-Parameter auf vorberechneten Scores."""
    total_gt = 0
    matched = 0
    total_triggers = 0
    total_frames = 0

    for video_data in video_scores.values():
        gate = EventGate(config)
        trigger_frames = []

        for frame_idx, score in video_data["scores"]:
            triggered = gate.check_event({"event_score": score}, frame_idx)
            if triggered is not False:
                trigger_frames.append(int(triggered))

        gt_events = video_data["gt_events"]
        used_triggers = set()
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers
                       and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)
                matched += 1

        total_gt += len(gt_events)
        total_triggers += len(trigger_frames)
        total_frames += video_data["total_frames"]

    recall = matched / total_gt if total_gt > 0 else 0
    precision = matched / total_triggers if total_triggers > 0 else 0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    call_red = 1 - (total_triggers / total_frames) if total_frames > 0 else 0

    return {
        "f1": round(f1, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "call_red": round(call_red, 4),
        "triggers": total_triggers,
    }


def _print_results(results: list, top_n: int):
    header = (f"{'Rank':<6}{'k':<7}{'k_min':<8}{'k_max':<8}{'cf':<6}"
              f"{'F1':<8}{'recall':<9}{'prec':<8}{'call_red'}")
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(results[:top_n], 1):
        print(f"{rank:<6}{r['k']:<7}{r['k_min']:<8}{r['k_max']:<8}{r['confirm_frames']:<6}"
              f"{r['f1']:<8}{r['recall']:<9}{r['precision']:<8}{r['call_red']}")


def tune_params(
    dataset: str,
    split: str,
    weights: str = None,
    use_mamba: bool = True,
    k_values: list = None,
    confirm_frames_values: list = None,
    k_min_ratio_values: list = None,
    k_max_ratio_values: list = None,
):
    if k_values is None:
        k_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    if confirm_frames_values is None:
        confirm_frames_values = [3, 5, 7, 10, 13, 15]
    if k_min_ratio_values is None:
        k_min_ratio_values = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    if k_max_ratio_values is None:
        k_max_ratio_values = [1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]

    config = load_config(dataset)
    video_ids = getattr(config, f"video_ids_{split}")

    print(f"Pre-computing event scores for {len(video_ids)} videos...")
    video_scores = _precompute_scores(config, dataset, video_ids, weights, use_mamba)
    print("Done.\n")

    # --- Stage 1: k / confirm_frames mit weiten Grenzen ---
    stage1_combos = list(itertools.product(k_values, confirm_frames_values))
    print(f"Stage 1: {len(stage1_combos)} Kombinationen (k × confirm_frames) ...")

    stage1_results = []
    for i, (k, cf) in enumerate(stage1_combos, 1):
        config.k = k
        config.k_min = round(k * 0.5, 4)
        config.k_max = round(k * 2.0, 4)
        config.confirm_frames = cf
        metrics = _eval_gate(config, video_scores)
        stage1_results.append({"k": k, "k_min": config.k_min, "k_max": config.k_max,
                                "confirm_frames": cf, **metrics})
        if i % 10 == 0 or i == len(stage1_combos):
            best = max(stage1_results, key=lambda x: x["f1"])
            print(f"  [{i}/{len(stage1_combos)}] best F1: {best['f1']:.4f}  "
                  f"(k={best['k']}, cf={best['confirm_frames']})")

    stage1_results.sort(key=lambda x: x["f1"], reverse=True)
    print("\nStage 1 top 5:")
    _print_results(stage1_results, 5)

    # --- Stage 2: k_min / k_max für die 5 besten Kandidaten ---
    boundary_combos = list(itertools.product(k_min_ratio_values, k_max_ratio_values))
    stage2_candidates = stage1_results[:5]
    total_stage2 = len(stage2_candidates) * len(boundary_combos)
    print(f"\nStage 2: {total_stage2} Kombinationen (5 Kandidaten × {len(boundary_combos)} Grenzen) ...")

    stage2_results = []
    for i, (candidate, (k_min_r, k_max_r)) in enumerate(
        itertools.product(stage2_candidates, boundary_combos), 1
    ):
        config.k = candidate["k"]
        config.k_min = round(candidate["k"] * k_min_r, 4)
        config.k_max = round(candidate["k"] * k_max_r, 4)
        config.confirm_frames = candidate["confirm_frames"]
        metrics = _eval_gate(config, video_scores)
        stage2_results.append({
            "k": candidate["k"],
            "k_min": config.k_min,
            "k_max": config.k_max,
            "confirm_frames": candidate["confirm_frames"],
            **metrics,
        })
        if i % 50 == 0 or i == total_stage2:
            best = max(stage2_results, key=lambda x: x["f1"])
            print(f"  [{i}/{total_stage2}] best F1: {best['f1']:.4f}  "
                  f"(k={best['k']}, k_min={best['k_min']}, k_max={best['k_max']}, cf={best['confirm_frames']})")

    stage2_results.sort(key=lambda x: x["f1"], reverse=True)
    print("\nFinal top 5:")
    _print_results(stage2_results, 5)

    return stage2_results[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",   choices=["epickitchen", "soccernet"])
    parser.add_argument("split",     choices=["train", "test"])
    parser.add_argument("--weights", default=None, help="Pfad zu trainierten Gewichten (.pt)")
    parser.add_argument("--no_mamba", action="store_true", help="Ablation: Mamba weglassen")
    args = parser.parse_args()

    best = tune_params(args.dataset, args.split, args.weights, use_mamba=not args.no_mamba)
    print(f"\nBest: k={best['k']}, k_min={best['k_min']}, k_max={best['k_max']}, "
          f"confirm_frames={best['confirm_frames']}  ->  F1={best['f1']}, recall={best['recall']}")
