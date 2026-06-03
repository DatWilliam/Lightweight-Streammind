from tqdm import tqdm
import argparse
from config import load_config
from model.epfe import EPFE
from model.gate import EventGate
from utils.eval_func import per_video_metrics, macro_average, count_phase_fp

BATCH_SIZE = 64


def run_eval(dataset: str, split: str):
    config = load_config(dataset)
    video_ids = getattr(config, f"video_ids_{split}")

    if dataset == "ego4d":
        from data.prepare_ego4d import load_video_labels, load_video, get_frame_count
    else:
        from data.prepare_soccernet import load_video_labels, load_video, get_frame_count

    per_video = []
    skipped_no_gt = 0

    for video_id in video_ids:
        gate = EventGate(config)
        epfe = EPFE(config)
        gt_events = load_video_labels(video_id)
        trigger_frames = []
        batch_frames, batch_indices = [], []
        n_frames = get_frame_count([video_id])

        for frame_idx, frame in enumerate(tqdm(load_video(video_id), total=n_frames, desc=video_id), start=1):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            if len(batch_frames) == BATCH_SIZE:
                for idx, feat in zip(batch_indices, epfe.process_batch(batch_frames)):
                    triggered = gate.check_event(feat, idx)
                    if triggered is not False:
                        trigger_frames.append(int(triggered))
                batch_frames, batch_indices = [], []

        if batch_frames:
            for idx, feat in zip(batch_indices, epfe.process_batch(batch_frames)):
                triggered = gate.check_event(feat, idx)
                if triggered is not False:
                    trigger_frames.append(int(triggered))

        used_triggers = set()
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)

        tp = len(used_triggers)
        false_triggers = [t for t in trigger_frames if t not in used_triggers]
        gt_starts = [e["start_frame"] for e in gt_events]
        fp_phase = count_phase_fp(false_triggers, gt_starts)

        v = per_video_metrics(tp, fp_phase, len(gt_events), len(trigger_frames), n_frames)
        if v is None:
            skipped_no_gt += 1
            continue
        per_video.append(v)

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
    parser.add_argument("split", choices=["train", "test"], default="train", nargs="?")
    args = parser.parse_args()
    run_eval(args.dataset, args.split)
