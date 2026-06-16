import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Generator, Optional
from config import load_config

cfg = load_config("ego4d")

VIDEO_DIR = Path("/mnt/hdd/liam_wipperfuerth/v2/video_540ss")

# caches always on SSD, regardless of video location
CACHE_DIR = cfg.DATA_DIR / "ego4d" / "v2" / "cache"
NARRATION_PATH = cfg.DATA_DIR / "ego4d" / "v2" / "annotations" / "narration.json"

_NARRATIONS: Optional[Dict[str, List[Dict]]] = None

def _load_narrations() -> Dict[str, List[Dict]]:
    # Load narration.json once and store in _NARRATIONS, keyed by UID
    # Following StreamMind Alg. 1
    #   Keep only #C narrations (camera wearer actions)
    #   Merge consecutive identical texts (keep first timestamp)
    #   Map timestamps to frames
    # If two events land on the same frame index, throw the later one away

    global _NARRATIONS
    if _NARRATIONS is not None:
        return _NARRATIONS

    if not NARRATION_PATH.exists():
        raise FileNotFoundError(f"narration.json not found: {NARRATION_PATH}")

    with open(NARRATION_PATH, encoding="utf-8") as f:
        data = json.load(f)

    buckets: Dict[str, List[Dict]] = {}
    for uid, val in data.items():
        if not isinstance(val, dict):
            continue

        # collect both passes, filter to #C, sort by timestamp
        raw = []
        for pass_key in ("narration_pass_1", "narration_pass_2"):
            pv = val.get(pass_key)
            if not isinstance(pv, dict):
                continue
            for n in pv.get("narrations", []):
                text = (n.get("narration_text", "") or "").strip()
                if not text.startswith("#C"):
                    continue
                raw.append({
                    "timestamp_sec": float(n.get("timestamp_sec", 0.0)),
                    "narration_text": text,
                    "pass": pass_key,
                })
        raw.sort(key=lambda r: r["timestamp_sec"])

        # dedupe consecutive identical texts (keep first timestamp)
        deduped = []
        prev_text = None
        for r in raw:
            if r["narration_text"] == prev_text:
                continue
            deduped.append(r)
            prev_text = r["narration_text"]

        # map timestamps to frame indices at cfg.fps; dedupe by frame
        events: Dict[int, Dict] = {}
        for r in deduped:
            start_frame = max(1, round(r["timestamp_sec"] * cfg.fps))
            if start_frame in events:
                continue
            events[start_frame] = {
                "start_frame": start_frame,
                "timestamp_sec": r["timestamp_sec"],
                "narration_text": r["narration_text"],
                "pass": r["pass"],
            }
        buckets[uid] = sorted(events.values(), key=lambda e: e["start_frame"])

    _NARRATIONS = buckets
    return _NARRATIONS


def load_video_labels(video_id: str) -> List[Dict]:
    # video_id = Ego4D UID (no extension)
    narrations = _load_narrations()
    if video_id not in narrations:
        raise KeyError(f"No narrations for video {video_id}")
    return narrations[video_id]


def get_video_path(video_id: str) -> Path:
    return VIDEO_DIR / f"{video_id}.mp4"


def get_cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.npz"


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
