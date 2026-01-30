from perception import Perception
from event_gate import EventGate
from utils import load_video
import matplotlib.pyplot as plt
import csv
import cv2

video_paths = [
    "../data/1.mp4"
    #"../data/act1.mp4",
    #"../data/act2.mp4",
    #"../data/act3.mp4",
    #"../data/act4.mp4",
    #"../data/act5.mp4",
    #"../data/act6.mp4",
]

perception = Perception()
gate = EventGate()

event_log = []

event_sum = 0
frame_idx = 0
video_idx = 0

for video_path in video_paths:
    print(f"\n=== Processing Video {video_idx+1}: {video_path} ===")

    for frame in load_video(video_path):
        features = perception.process_frame(frame)  # Perception Token

        event_log.append({
            "frame": frame_idx,
            "video": video_idx,
            "event_score": features["event_score"]
        })

        # Debug: Motion & Scene Change Score
        print(
            f"Frame {frame_idx} -> "
            f"Event Score: {features['event_score']:.4f}"
        )

        # Cognition / Event Gate
        if gate.check_event(features, frame_idx):
            cv2.imwrite(
                f"../data/video{video_idx}_frame{frame_idx}.jpg",
                frame
            )
            event_sum += 1
            print(
                f"*** EVENT DETECTED *** "
                f"Frame {frame_idx} (Video {video_idx}) | "
                f"Event Score: {features['event_score']:.4f}"
            )

        frame_idx += 1

    video_idx += 1

print(f"\n=== Processed Video Summary ===")
print(f"Total Events: {event_sum}")

with open("event_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["video", "frame", "event_score"]
    )
    writer.writeheader()
    writer.writerows(event_log)

frames = [row["frame"] for row in event_log]
scores = [row["event_score"] for row in event_log]

plt.figure()
plt.plot(frames, scores)
plt.xlabel("Frame")
plt.ylabel("Event Score")
plt.title("EPFE Event Score over Time")
plt.show()