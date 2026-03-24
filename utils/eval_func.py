def calculate_timval(llm_calls: int, detected_events: int, gt_events: int) -> dict:
    TP = detected_events
    FP = llm_calls - detected_events
    FN = gt_events - detected_events
    total_silence = gt_events + 1
    precision = TP / llm_calls if llm_calls > 0 else 0.0
    recall = TP / gt_events if gt_events > 0 else 0.0
    timval = (1 - FP / total_silence) * (1 - FN / gt_events) if gt_events > 0 else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "precision": precision, "recall": recall, "timval": round(timval, 3)}


def calculate_triggeracc(llm_calls: int, detected_events: int, gt_events: int) -> float:
    TP = detected_events
    FP = llm_calls - TP
    FN = gt_events - TP
    TN = (gt_events + 1) - FP
    return round((TP + TN) / (TP + FP + TN + FN), 3)