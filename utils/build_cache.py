# python -m utils.build_cache
import numpy as np
from tqdm import tqdm
from pathlib import Path
import clip
import torch
from PIL import Image
import cv2
from config import load_config

BATCH_SIZE = 16


def extract_features(video_path: Path, model, preprocess, device) -> tuple:
    frame_indices, features = [], []
    batch_frames, batch_indices = [], []

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(video_path))
    for frame_idx in tqdm(range(1, total + 1), desc=video_path.name):
        ret, frame = cap.read()
        if not ret:
            break
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
    cap.release()

    # letzter Batch
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

    return frame_indices, features


def build_cache():
    config = load_config("soccernet")
    soccernet_dir = config.DATA_DIR / "soccernet"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(config.clip_model, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # alle .mkv dateien rekursiv finden
    video_files = sorted(soccernet_dir.rglob("*.mkv"))
    print(f"Gefunden: {len(video_files)} Videos")

    for video_path in video_files:
        cache_path = video_path.with_suffix(".npz")
        if cache_path.exists():
            print(f"Already cached: {video_path.name}, skipping")
            continue

        print(f"\nVerarbeite: {video_path}")
        frame_indices, features = extract_features(video_path, model, preprocess, device)

        np.savez(cache_path,
                 frame_idx=np.array(frame_indices),
                 features=np.array(features, dtype=np.float32))
        print(f"Saved {cache_path} ({len(frame_indices)} frames)")


if __name__ == "__main__":
    build_cache()