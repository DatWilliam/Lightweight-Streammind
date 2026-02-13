from tqdm import tqdm
from config import cfg
from model.epfe import EPFE
from model.gate import EventGate
from data.prepare_epickitchen import load_video, get_frame_count

print("Loading CLIP model...")
epfe = EPFE()
gate = EventGate()
print("Model loaded.")

video_id = "P01_102"
frame_idx = 0

for i, frame in enumerate(tqdm(load_video(video_id), total=get_frame_count(video_id))):
    frame_idx += 1
    if i % cfg.frame_skip != 0:
        continue

    features = epfe.process_frame(frame)

    if gate.check_event(features, frame_idx):
        print("event detected")