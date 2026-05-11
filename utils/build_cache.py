import sys
import numpy as np
from tqdm import tqdm
from pathlib import Path
import clip
import torch
from PIL import Image
import cv2
from config import load_config

BATCH_SIZE = 16

DATASETS = ("soccernet", "ego4d")


def _paths(dataset: str, video_id: str) -> tuple:
    if dataset == "soccernet":
        from data.prepare_soccernet import get_video_path, get_cache_path
    elif dataset == "ego4d":
        from data.prepare_ego4d import get_video_path, get_cache_path
    else:
        raise ValueError(f"Unbekanntes Dataset: {dataset}")
    return get_video_path(video_id), get_cache_path(video_id)


def extract_features(video_path: Path, model, preprocess, device, sample_stride: int = 1) -> tuple:
    """
    Extract per-frame CLIP features.
      sample_stride=1  -> every frame
      sample_stride=15 -> every 15th frame (30 fps -> 2 fps)
    Output frame_idx is 1-indexed at the target fps (1, 2, 3, ...).
    """
    frame_indices, features = [], []
    batch_frames, batch_indices = [], []

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cap = cv2.VideoCapture(str(video_path))
    out_idx = 0
    for src_idx in tqdm(range(total), desc=video_path.name):
        ret, frame = cap.read()
        if not ret:
            break
        if src_idx % sample_stride != 0:
            continue
        out_idx += 1
        batch_frames.append(frame)
        batch_indices.append(out_idx)

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


def _collect_video_ids(cfg) -> list:
    # union of train/val/test ids, dedupe while preserving order
    ids = []
    for split in ("train", "val", "test"):
        ids.extend(getattr(cfg, f"video_ids_{split}", []))
    seen = set()
    unique = []
    for vid in ids:
        if vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique


def build_cache_for_dataset(dataset: str, model, preprocess, device) -> None:
    print(f"\n=== {dataset} ===")
    cfg = load_config(dataset)
    video_ids = _collect_video_ids(cfg)
    if not video_ids:
        print(f"No video_ids_{{train,val,test}} configured, skipping.")
        return

    def _pretty(p: Path) -> str:
        try:
            return str(p.relative_to(cfg.DATA_DIR))
        except ValueError:
            return str(p)

    for video_id in video_ids:
        video_path, cache_path = _paths(dataset, video_id)

        if cache_path.exists():
            print(f"[skip] {_pretty(cache_path)}")
            continue

        if not video_path.exists():
            print(f"[miss] video not found: {video_path}")
            continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stride = getattr(cfg, "sample_stride", 1)
        print(f"\nProcessing: {_pretty(video_path)}  (stride={stride})")
        frame_indices, features = extract_features(video_path, model, preprocess, device, sample_stride=stride)

        np.savez(
            cache_path,
            frame_idx=np.array(frame_indices),
            features=np.array(features, dtype=np.float32),
        )
        print(f"Saved: {_pretty(cache_path)} ({len(frame_indices)} frames)")


def build_cache(datasets=None) -> None:
    datasets = datasets or DATASETS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = load_config("soccernet").clip_model  # same across datasets via BaseConfig
    print(f"Device: {device} | CLIP: {clip_model}")
    model, preprocess = clip.load(clip_model, device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for ds in datasets:
        build_cache_for_dataset(ds, model, preprocess, device)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        unknown = [a for a in args if a not in DATASETS]
        if unknown:
            print(f"Unknown datasets: {unknown}. Allowed: {DATASETS}")
            sys.exit(1)
        build_cache(args)
    else:
        build_cache()
