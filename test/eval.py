import json
from tqdm import tqdm
from src.perception import Perception
from src.event_gate import EventGate
from src.utils import load_video


video_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
label_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels.json"

# Return arr with all 'ground truth' event frames
def load_gt_events(half):
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []
    for ann in data.get("annotations", []):
        if str(ann.get("half")) == str(half):
            events.append(ann["frame"])

    return sorted(events)

perception = Perception()
gate = EventGate()

SKIP_RATE = 3
TOLERANCE = 50

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

print("\n=== Event Gate Evaluation ===")
print(f"Total GT Events:     {len(gt_events)}")
print(f"Detected Events:     {len(gt_matched)}")
print(f"Event Recall:        {event_recall:.2%}")
print(f"Average Delay:       {avg_delay:.1f} frames")
print(f"LLM Calls (Triggers):{llm_calls}")
print(f"Reduction vs per-step: {(1 - llm_calls / total_frames) * 100:.1f}%")
