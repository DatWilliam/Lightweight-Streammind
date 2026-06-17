import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Generator
from config import load_config

cfg = load_config("soccernet")


# read the events of one half from Labels-v2.json (positions are ms since kickoff)
def load_video_labels(video_id: str) -> List[Dict]:
    video_path = cfg.DATA_DIR / "soccernet" / video_id
    json_path = video_path.parent / "Labels-v2.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Labels-v2.json not found: {json_path}")

    # "1_224p.mkv" -> half "1", "2_224p.mkv" -> half "2"
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


def get_video_path(video_id: str) -> Path:
    return cfg.DATA_DIR / "soccernet" / video_id


def get_cache_path(video_id: str) -> Path:
    return get_video_path(video_id).with_suffix(".npz")


def get_total_gt_events(video_ids: List[str]) -> int:
    return sum(len(load_video_labels(vid)) for vid in video_ids)


def get_frame_count(video_ids: List[str]) -> int:
    # read from cache; fall back to video if cache missing
    total = 0
    for video_id in video_ids:
        cache_path = get_cache_path(video_id)
        if cache_path.exists():
            with np.load(str(cache_path)) as data:
                total += len(data["frame_idx"])
        else:
            video_path = get_video_path(video_id)
            cap = cv2.VideoCapture(str(video_path))
            total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    return total


def load_video(video_id: str) -> Generator[np.ndarray, None, None]:
    # used by the live (non-cached) inference path
    video_path = get_video_path(video_id)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
