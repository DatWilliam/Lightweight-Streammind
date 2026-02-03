import json

from perception import Perception
from event_gate import EventGate
from utils import load_video
import cv2

video_paths = [
    "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
]

perception = Perception()
gate = EventGate()

debug = False
SKIP_RATE = 5

test_data_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels.json"

event_sum = 0
frame_idx = 0
video_idx = 1

for video_path in video_paths:
    print(f"\n=== Processing Video {video_idx}: {video_path} ===")

    for i, frame in enumerate(load_video(video_path)):
        frame_idx += 1

        if i % SKIP_RATE != 0:
            continue

        features = perception.process_frame(frame)

        # Cognition Gate (Silence/Response)
        if gate.check_event(features, frame_idx):

            description = perception.label_frame(frame)
            print(
                f"*** EVENT DETECTED *** "
                f"Frame {frame_idx} (Video {video_idx}) | "
                f"Scene: {description[0]['label']} ({description[0]['confidence']}) | "
                f""f"Event Score: {features['event_score']:.4f}"
            )

            if debug:
                cv2.imwrite(f"../data/video{video_idx}_frame{frame_idx}_{description[0]['label']}.jpg",frame)
            event_sum += 1

        else :
            print(f"Frame {frame_idx} -> "f"Event Score: {features['event_score']:.4f}")

    video_idx += 1

print(f"\n=== Processed Video Summary ===")
print(f"Total Events: {event_sum}")
