from tqdm import tqdm
from src.perception import Perception
from src.event_gate import EventGate
from src.utils import load_video

video_path = "../data/soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"

perception = Perception()
gate = EventGate()

SKIP_RATE = 3

frame_idx = 0
video_idx = 1

for i, frame in tqdm(list(enumerate(load_video(video_path)))):
    frame_idx += 1
    if i % SKIP_RATE != 0:
        continue

    features = perception.process_frame(frame)

    if gate.check_event(features, frame_idx):
        print("event detected")