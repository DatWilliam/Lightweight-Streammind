from tqdm import tqdm
from config import cfg
from model.epfe import EPFE
from data.prepare_epickitchen import load_video, get_frame_count

print("Loading model...")
epfe = EPFE()
epfe.load_checkpoint("checkpoints/best.pt")
epfe.eval()
print("Model loaded.")

video_id = "P01_102"
frame_idx = 0
last_event_frame = -cfg.cooldown

for i, frame in enumerate(tqdm(load_video(video_id), total=get_frame_count(video_id))):
    frame_idx += 1
    if i % cfg.frame_skip != 0:
        continue

    result = epfe.process_frame(frame)

    if (
        result["event_score"] > cfg.event_threshold
        and frame_idx - last_event_frame >= cfg.cooldown
    ):
        print(f"Event detected at frame {frame_idx} (score: {result['event_score']:.3f})")
        last_event_frame = frame_idx