import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from model.epfe import EPFE
from model.gate import EventGate
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

video_ids = ["P01_102", "P01_103", "P01_104", "P01_106"]

make_plot = False

for video_id in video_ids:
    # new — reset models between videos so state doesn't carry over
    perception = EPFE()
    gate = EventGate()

    frame_idx = 0
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

        features = perception.process_frame(frame)

        # Store data for plotting
        frame_indices.append(frame_idx)
        event_scores.append(features["event_score"])

        if gate.check_event(features, frame_idx):
            trigger_frames.append(frame_idx)

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
        "model": cfg.clip_model,
        "frame_skip": cfg.frame_skip,
        "alpha": cfg.alpha,
        "window_size": cfg.window_size,
        "k": cfg.k,
        "cooldown": cfg.cooldown,
        "confirm_frames": cfg.confirm_frames
    }

    # Metrics
    event_recall = len(gt_matched) / len(gt_events)
    avg_delay = sum(delays) / len(delays) if delays else float("inf")
    llm_calls = len(trigger_frames)
    total_frames = frame_idx // cfg.frame_skip
    event_precision = len(gt_matched) / len(trigger_frames)
    f1_score = 2*((event_recall*event_precision) / (event_recall+event_precision))

    metrics = {
        "F1_SCORE": round(f1_score,3),
        "TOTAL_llm_calls": llm_calls,
        "TOTAL_detected_events": len(gt_matched),
        "TOTAL_gt_events": len(gt_events),
        "recall_percent": round(event_recall*100, 1),
        "precision_percent": round(event_precision*100, 1),
        "average_delay": round(avg_delay,1),
        "note": "confirm frames 3 -> 4"
    }

    # Save results
    save_results(PARAMS, metrics)
    print("\n✓ Results saved to results.json")

# === PLOT GENERATION ===
if make_plot:
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot Event Score
    ax.plot(frame_indices, event_scores, label='Event Score', color='blue', linewidth=1)

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
    ax.set_title('Event Detection Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('event_score_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved to event_score_analysis.png")
    plt.show()

