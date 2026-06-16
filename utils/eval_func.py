import bisect

# amount of distinct silence phases with a least one false trigger
def count_phase_fp(false_triggers, gt_start_frames) -> int:
    if not false_triggers:
        return 0
    if not gt_start_frames:
        return 1
    sorted_gts = sorted(gt_start_frames)
    return len({bisect.bisect_left(sorted_gts, t) for t in false_triggers})


def per_video_metrics(tp: int, fp_phase: int, gt_events: int,
                      n_triggers: int, n_frames: int) -> dict:
    if gt_events == 0:
        return None
    fn = gt_events - tp
    n_phases = gt_events + 1
    precision = tp / n_triggers if n_triggers > 0 else 0.0
    recall = tp / gt_events
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    call_red = 1 - (n_triggers / n_frames) if n_frames > 0 else 0.0
    timval = (1 - fp_phase / n_phases) * (1 - fn / gt_events)
    tn = n_phases - fp_phase
    trigger_acc = (tp + tn) / (tp + fp_phase + tn + fn)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "call_red": call_red,
        "tim_val": max(0.0, timval),
        "trigger_acc": max(0.0, trigger_acc),
    }


def macro_average(per_video: list) -> dict:
    if not per_video:
        return {k: 0.0 for k in ("f1", "precision", "recall", "call_red", "tim_val", "trigger_acc")}
    n = len(per_video)
    keys = per_video[0].keys()
    return {k: sum(v[k] for v in per_video) / n for k in keys}


# for micro avg (not used)
def calculate_timval(llm_calls: int, detected_events: int, gt_events: int,
                     fp: int = None) -> dict:
    TP = detected_events
    FP = (llm_calls - detected_events) if fp is None else fp
    FN = gt_events - detected_events
    total_silence = gt_events + 1
    precision = TP / llm_calls if llm_calls > 0 else 0.0
    recall = TP / gt_events if gt_events > 0 else 0.0
    timval = (1 - FP / total_silence) * (1 - FN / gt_events) if gt_events > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "timval": round(timval, 3)}


def calculate_triggeracc(llm_calls: int, detected_events: int, gt_events: int,
                         fp: int = None) -> float:
    TP = detected_events
    FP = (llm_calls - TP) if fp is None else fp
    FN = gt_events - TP
    TN = (gt_events + 1) - FP
    return round((TP + TN) / (TP + FP + TN + FN), 3)
