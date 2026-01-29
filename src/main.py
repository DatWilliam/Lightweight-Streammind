from perception import Perception
from event_gate import EventGate
from utils import load_video

video_paths = [
    "../data/act1.mp4",
    "../data/act2.mp4",
    "../data/act3.mp4",
    "../data/act4.mp4",
    "../data/act5.mp4",
    "../data/act6.mp4",
]

perception = Perception()
gate = EventGate()

event_sum = 0

frame_idx = 0
video_idx = 0

for video_path in video_paths:
    print(f"\n=== Processing Video {video_idx+1}: {video_path} ===")
    for frame in load_video(video_path):
        features = perception.process_frame(frame)  # Perception Token

        # Debug: Motion & Scene Change Score
        print(f"Frame {frame_idx} -> Motion: {features['motion_score']:.4f}, Scene Change: {features['scene_change_score']:.4f}")

        # Cognition / Event Gate
        if gate.check_event(features, frame_idx):
            event_sum += 1
            print(f"*** EVENT DETECTED, Frame {frame_idx} (Video {video_idx}) *** | Features: {features}")

        frame_idx += 1

    video_idx += 1

print(f"\n=== Processed Video Summary ===")
print(f"Total Events: {event_sum}")