# python -m utils.build_cache epickitchen
# python -m utils.build_cache soccernet
import os
import argparse
import numpy as np
from tqdm import tqdm
import clip
import torch
from PIL import Image
import cv2
from config import load_config

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH_SIZE = 64


def build_cache(dataset: str):
    if dataset == "epickitchen":
        from data.prepare_epickitchen import load_video, get_frame_count
    else:
        from data.prepare_soccernet import load_video, get_frame_count

    config = load_config(dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.clip_model, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    all_ids = config.video_ids_train + config.video_ids_test

    for video_id in all_ids:
        cache_path = os.path.join(CACHE_DIR, f"{video_id}.npz")
        if os.path.exists(cache_path):
            print(f"Already cached: {video_id}, skipping")
            continue

        frame_indices = []
        features = []
        batch_frames = []
        batch_indices = []

        for frame_idx, frame in enumerate(tqdm(load_video(video_id), total=get_frame_count([video_id]), desc=video_id), start=1):
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

            if len(batch_frames) == BATCH_SIZE:
                images = torch.stack([
                    preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                    for f in batch_frames
                ]).to(device)
                with torch.no_grad():
                    feats = model.encode_image(images)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    feats = feats.cpu().numpy().astype(np.float32)
                frame_indices.extend(batch_indices)
                features.extend(feats)
                batch_frames, batch_indices = [], []

        if batch_frames:
            images = torch.stack([
                preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
                for f in batch_frames
            ]).to(device)
            with torch.no_grad():
                feats = model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats = feats.cpu().numpy().astype(np.float32)
            frame_indices.extend(batch_indices)
            features.extend(feats)

        np.savez(cache_path,
                 frame_idx=np.array(frame_indices),
                 features=np.array(features, dtype=np.float32))
        print(f"Saved {cache_path} ({len(frame_indices)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["epickitchen", "soccernet"])
    args = parser.parse_args()
    build_cache(args.dataset)