import cv2
import pandas as pd
from config import cfg
from pathlib import Path
import numpy as np
from typing import List, Dict, Generator

# Load all events for given video
def load_video_labels(video_id: str) -> List[Dict]:
    csv_path = Path(cfg.ann_dir) / "EPIC_100_train.csv"
    dataframe = pd.read_csv(csv_path)

    # take all data with corresponding video id
    video_dataframe = dataframe[dataframe["video_id"] == video_id].copy()
    video_dataframe = video_dataframe.sort_values("start_frame")

    events = []
    for _, row in video_dataframe.iterrows():
        events.append({
            "start_frame": int(row["start_frame"]),
            "stop_frame": int(row["stop_frame"]),
            "narration": row["narration"],
            "verb": row["verb"],
            "noun": row["noun"]
        })

    return events

def get_event_start_frames(video_id: str) -> List[int]:
    events = load_video_labels(video_id)
    return [event["start_frame"] for event in events]

def get_frame_count(video_id: str) -> int:
    video_path = Path(cfg.epickitchen_video_root) / "videos" / f"{video_id}.MP4"
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total

def load_video(video_id: str) -> Generator[np.ndarray, None, None]:
    participant = video_id.split("_")[0]
    video_path = Path(cfg.epickitchen_video_root) / "videos" / f"{video_id}.MP4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video {video_id} does not exist. Path: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Video {video_id} could not be opened")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_count >= total_frames - 10:
                break
            else:
                print(f"Warning: Video ended early at frame {frame_count}/{total_frames}")
                break
        yield frame
        frame_count += 1

    cap.release()