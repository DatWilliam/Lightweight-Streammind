import json
from tqdm import tqdm
from src.perception import Perception
from src.event_gate import EventGate
from src.utils import load_video


video_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
label_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels.json"


def save_results(params, metrics, filename="results.json"):
    result = {
        "parameters": params,
        "metrics": metrics
    }

    try:
        with open(filename, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    results.append(result)

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

PARAMS = {
    "video_name": "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley : 1st half",
    "model": "ViT-B/32",
    "frame_skip": 2,
    "alpha": 0.3,
    "window_size": 60,
    "k": 2.0,
    "cooldown": 45,
    "tolerance": 35
}

# Return arr with all 'ground truth' event frames
def load_gt_events(half):
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []
    for ann in data.get("annotations", []):
        if str(ann.get("half")) == str(half):
            events.append(ann["frame"])

    return sorted(events)

perception = Perception(PARAMS["alpha"])
gate = EventGate(PARAMS["window_size"], PARAMS["k"], PARAMS["cooldown"])

SKIP_RATE = PARAMS["frame_skip"]
TOLERANCE = PARAMS["tolerance"]

frame_idx = 0
video_idx = 1

gt_events = load_gt_events(half=video_idx)
gt_matched = set()  # allows no duplicates

trigger_frames = [] # event frames
delays = []         # to see avg delay

for i, frame in tqdm(list(enumerate(load_video(video_path)))):
    frame_idx += 1
    if i % SKIP_RATE != 0:
        continue

    features = perception.process_frame(frame)

    if gate.check_event(features, frame_idx):
        trigger_frames.append(frame_idx)  # if gate gave response token: save frame idx

# Check for matches between trigger frames and gt events
for gt in gt_events:
    matches = []
    for t in trigger_frames:
        if abs(t - gt) <= TOLERANCE:
            matches.append(t)

    if matches:
        closest = min(matches, key=lambda x: abs(x - gt)) # find trigger frame closest to gt
        gt_matched.add(gt)
        delays.append(closest - gt)

# Metrics
event_recall = len(gt_matched) / len(gt_events)
avg_delay = sum(delays) / len(delays) if delays else float("inf")
llm_calls = len(trigger_frames)
total_frames = frame_idx // SKIP_RATE

metrics = {
    "total_gt_events": len(gt_events),
    "detected_events": len(gt_matched),
    "event_recall": event_recall,
    "average_delay": avg_delay,
    "llm_calls": llm_calls,
    "reduction_percent": (1 - llm_calls / total_frames) * 100
}

print("\n=== Event Gate Evaluation ===")
print(f"Total GT Events:     {metrics['total_gt_events']}")
print(f"Detected Events:     {metrics['detected_events']}")
print(f"Event Recall:        {metrics['event_recall']:.2%}")
print(f"Average Delay:       {metrics['average_delay']:.1f} frames")
print(f"LLM Calls (Triggers):{metrics['llm_calls']}")
print(f"Reduction vs per-step: {metrics['reduction_percent']:.1f}%")

# Save results
save_results(PARAMS, metrics)
print("\n✓ Results saved to results.json")
