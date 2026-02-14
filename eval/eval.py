import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.epfe import EPFE
from config import cfg
from data.prepare_epickitchen import load_video_labels, load_video, get_frame_count

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

video_id = "P01_106"

perception = EPFE()
perception.load_checkpoint("checkpoints/best.pt")
perception.eval()

frame_idx = 0
last_event_frame = -cfg.cooldown

make_plot = False

gt_events = load_video_labels(video_id)
gt_matched = set()  # allows no duplicates

trigger_frames = []  # event frames
delays = []  # to see avg delay

# Store data for plotting
frame_indices = []
event_scores = []

for i, frame in enumerate(tqdm(load_video(video_id), total=get_frame_count(video_id))):
    frame_idx += 1
    if i % cfg.frame_skip != 0:
        continue

    result = perception.process_frame(frame)

    # Store data for plotting
    frame_indices.append(frame_idx)
    event_scores.append(result["event_score"])

    if (
        result["event_score"] > cfg.event_threshold
        and frame_idx - last_event_frame >= cfg.cooldown
    ):
        trigger_frames.append(frame_idx)
        last_event_frame = frame_idx

# Check for matches: adaptive tolerance + 1-to-1 matching
used_triggers = set()
for gt in gt_events:
    duration = gt["stop_frame"] - gt["start_frame"]
    tolerance = min(duration * cfg.tolerance_fraction, cfg.max_tolerance)

    matches = []
    for t in trigger_frames:
        if t not in used_triggers and abs(t - gt["start_frame"]) <= tolerance:
            matches.append(t)

    if matches:
        closest = min(matches, key=lambda x: abs(x - gt["start_frame"]))
        used_triggers.add(closest)
        gt_matched.add(gt["start_frame"])
        delays.append(closest - gt["start_frame"])

PARAMS = {
    "video_name": video_id,
    "model": f"CLIP({cfg.clip_model}) + Mamba(layers={cfg.n_mamba_layers})",
    "frame_skip": cfg.frame_skip,
    "d_model": cfg.d_model,
    "n_mamba_layers": cfg.n_mamba_layers,
    "d_state": cfg.d_state,
    "event_threshold": cfg.event_threshold,
    "cooldown": cfg.cooldown,
}

# Metrics
event_recall = len(gt_matched) / len(gt_events)
avg_delay = sum(delays) / len(delays) if delays else float("inf")
llm_calls = len(trigger_frames)
total_frames = frame_idx // cfg.frame_skip
event_rate = len(gt_matched) / len(trigger_frames) if trigger_frames else 0.0

metrics = {
    "#_llm_calls": llm_calls,
    "#_detected_events": len(gt_matched),
    "total_gt_events": len(gt_events),
    "event_recall_%": round(event_recall*100, 1),
    "event_rate_%": round(event_rate*100, 1),
    "average_delay": round(avg_delay,1),
    "reduction_%": round((1 - llm_calls / total_frames) * 100, 1),
    "note": ""
}

# Save results
save_results(PARAMS, metrics)
print("\nResults saved to results.json")

# === PLOT GENERATION ===
if make_plot:
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot Event Score
    ax.plot(frame_indices, event_scores, label='Event Score', color='blue', linewidth=1)

    # Threshold line
    ax.axhline(y=cfg.event_threshold, color='orange', linestyle='--', alpha=0.7, label='Threshold')

    # Mark ground truth events
    for gt_frame in gt_events:
        ax.axvline(x=gt_frame["start_frame"], color='green', alpha=0.5, linestyle='-', linewidth=2)

    # Add single legend entry for GT events
    ax.axvline(x=-1, color='green', alpha=0.5, linestyle='-', linewidth=2, label='GT Events')

    # Mark trigger frames
    for trigger_frame in trigger_frames:
        ax.axvline(x=trigger_frame, color='red', alpha=0.3, linestyle='-', linewidth=2)

    # Add single legend entry for triggers
    if trigger_frames:
        ax.axvline(x=-1, color='red', alpha=0.3, linestyle='-', linewidth=1, label='Triggers')

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Event Score', fontsize=12)
    ax.set_title('Event Detection Analysis (CLIP + Mamba)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('event_score_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved to event_score_analysis.png")
    plt.show()