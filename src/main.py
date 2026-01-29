from perception import Perception
from event_gate import EventGate
from utils import load_video

video_path = "../data/1.mp4"

perception = Perception()
gate = EventGate()

for idx, frame in enumerate(load_video(video_path)):
    features = perception.process_frame(frame) # Perception Token

    # Debug
    print(f"Frame {idx} -> Motion: {features['motion_score']:.4f}, Scene Change: {features['scene_change_score']:.4f}")

    # Cognition Gate
    if gate.check_event(features, idx):
        print(f"*** EVENT DETECTED, Frame {idx} *** | Features: {features}")
