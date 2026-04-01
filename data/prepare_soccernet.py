import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Generator
from config import load_config

cfg = load_config("soccernet")


def load_video_labels(video_id: str) -> List[Dict]:
    """
    Laedt Events fuer ein Video aus Labels-v2.json.

    video_id: relativer Pfad ab data/soccernet/, z.B.
              "england_epl/2015-2016/2016-03-19 - 18-00 Chelsea 2 - 2 West Ham/1_224p.mkv"

    position in der JSON ist in Millisekunden ab Halbzeitbeginn.
    """
    video_path = cfg.DATA_DIR / "soccernet" / video_id
    json_path = video_path.parent / "Labels-v2.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Labels-v2.json nicht gefunden: {json_path}")

    # "1_224p.mkv" -> half = "1", "2_224p.mkv" -> half = "2"
    half = video_path.stem.split("_")[0]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    events = []
    for ann in data["annotations"]:
        if not ann["gameTime"].startswith(f"{half} - "):
            continue
        position_ms = int(ann["position"])
        start_frame = max(1, round(position_ms * cfg.fps / 1000))
        events.append({
            "start_frame": start_frame,
            "label": ann["label"],
            "team": ann["team"],
            "gameTime": ann["gameTime"],
        })

    events.sort(key=lambda e: e["start_frame"])
    return events


def get_event_start_frames(video_id: str) -> List[int]:
    return [e["start_frame"] for e in load_video_labels(video_id)]


def get_total_gt_events(video_ids: List[str]) -> int:
    return sum(len(load_video_labels(vid)) for vid in video_ids)


def get_frame_count(video_ids: List[str]) -> int:
    total = 0
    for video_id in video_ids:
        video_path = cfg.DATA_DIR / "soccernet" / video_id
        cap = cv2.VideoCapture(str(video_path))
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return total


def load_video(video_id: str) -> Generator[np.ndarray, None, None]:
    video_path = cfg.DATA_DIR / "soccernet" / video_id

    if not video_path.exists():
        raise FileNotFoundError(f"Video nicht gefunden: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
