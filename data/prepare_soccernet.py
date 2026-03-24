import cv2
import pandas as pd
import numpy as np
from typing import List, Dict, Generator
from config import load_config

cfg = load_config("soccernet")

# Load all video labels into a list
def load_video_labels(video_id: str) -> List[Dict]:
    csv_path = cfg.DATA_DIR / "soccernet" / "annotation" / "soccer_ann.csv"
    dataframe = pd.read_csv(csv_path)

    # take all data with corresponding video id
    video_dataframe = dataframe[dataframe["video_id"] == video_id].copy()
    video_dataframe = video_dataframe.sort_values("start_frame")

    events = []
    for _, row in video_dataframe.iterrows():
        events.append({
            "start_frame": int(row["start_frame"]),
            "label": row["label"],
            "team": row["team"],
            "gameTime": row["gameTime"]
        })

    return events

def get_event_start_frames(video_id: str) -> List[int]:
    events = load_video_labels(video_id)
    return [event["start_frame"] for event in events]

def get_total_gt_events(video_ids: List[str]) -> int:
    csv_path = cfg.DATA_DIR / "soccernet" / "annotation" / "soccer_ann.csv"
    dataframe = pd.read_csv(csv_path)
    return int(dataframe[dataframe["video_id"].isin(video_ids)].shape[0])

def get_frame_count(video_ids: List[str]) -> int:
    total = 0
    for video_id in video_ids:
        video_path = cfg.DATA_DIR / "soccernet" / "videos" / video_id
        cap = cv2.VideoCapture(str(video_path))
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return total

def load_video(video_id: str) -> Generator[np.ndarray, None, None]:
    video_path = cfg.DATA_DIR / "soccernet" / "videos" / video_id

    if not video_path.exists():
        raise FileNotFoundError(f"Video {video_id} does not exist. Path: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Video {video_id} could not be opened")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()