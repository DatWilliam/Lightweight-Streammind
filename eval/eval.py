# python -m eval.eval epickitchen test
# python -m eval.eval soccernet test
from tqdm import tqdm
import argparse
from config import load_config
from model.epfe import EPFE
from model.gate import EventGate
from utils.eval_func import calculate_timval, calculate_triggeracc

BATCH_SIZE = 64

def run_eval(dataset: str, split: str):

    config = load_config(dataset) # load config for dataset
    video_ids = getattr(config, f"video_ids_{split}")

    if dataset == "epickitchen":
        from data.prepare_epickitchen import load_video_labels, load_video, get_frame_count, get_total_gt_events
    else:
        from data.prepare_soccernet import load_video_labels, load_video, get_frame_count, get_total_gt_events

    all_trigger_frames = [] # over all videos
    found_events = 0

    for video_id in video_ids:
        gate = EventGate(config)
        epfe = EPFE(config)
        gt_events = load_video_labels(video_id)
        trigger_frames = [] # for current video
        batch_frames = []
        batch_indices = []

        for frame_idx, frame in enumerate(tqdm(load_video(video_id), total=get_frame_count([video_id]), desc=video_id), start=1):
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
        # look for closest trigger frame to gt
        # track so no double triggers
        for gt in gt_events:
            matches = [t for t in trigger_frames
                       if t not in used_triggers and abs(t - gt["start_frame"]) <= config.tolerance * config.fps]
            if matches:
                closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
                used_triggers.add(closest)
                found_events += 1

        all_trigger_frames.extend(trigger_frames)

    llm_calls = len(all_trigger_frames)
    call_red = 1 - (llm_calls / get_frame_count(video_ids))
    recall = found_events / get_total_gt_events(video_ids)
    precision = found_events / llm_calls
    f1 = 2 * recall * precision / (recall + precision)

    metrics = {
        "F1_SCORE": round(f1, 3),
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "trigger_acc": max(0.0, calculate_triggeracc(llm_calls, found_events, get_total_gt_events(video_ids))),
        "tim_val": max(0.0, calculate_timval(llm_calls, found_events, get_total_gt_events(video_ids))["timval"]),
        "call_red_percent": round(call_red * 100, 2),
    }

    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["epickitchen", "soccernet"], default="epickitchen", nargs="?")
    parser.add_argument("split", choices=["train", "test"], default="train", nargs="?")
    args = parser.parse_args()
    run_eval(args.dataset, args.split)