import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import cfg
from data.prepare_epickitchen import load_video, load_video_labels, get_frame_count


def get_training_video_ids():
    """Get all video IDs from the training annotations."""
    import pandas as pd
    csv_path = Path(cfg.ann_dir) / "EPIC_100_train.csv"
    df = pd.read_csv(csv_path)
    return sorted(df["video_id"].unique().tolist())


def get_validation_video_ids():
    """Get all video IDs from the validation annotations."""
    import pandas as pd
    csv_path = Path(cfg.ann_dir) / "EPIC_100_validation.csv"
    df = pd.read_csv(csv_path)
    return sorted(df["video_id"].unique().tolist())


def extract_features(video_ids, output_dir="data/features"):
    """Pre-extract and cache CLIP features for a list of videos.

    Saves per-video .pt files containing (N, d_model) feature tensors
    where N = number of frames after frame_skip.
    """
    import clip
    from PIL import Image
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(cfg.clip_model, device=device)
    model.eval()

    for video_id in video_ids:
        out_path = output_dir / f"{video_id}.pt"
        if out_path.exists():
            print(f"  Skipping {video_id} (already extracted)")
            continue

        print(f"  Extracting {video_id}...")
        features = []
        total = get_frame_count(video_id)

        for i, frame in enumerate(tqdm(load_video(video_id), total=total, desc=video_id)):
            if i % cfg.frame_skip != 0:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model.encode_image(image)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                features.append(feature.squeeze(0).cpu())

        features = torch.stack(features)  # (N, d_model)
        torch.save(features, out_path)
        print(f"  Saved {video_id}: {features.shape}")


def make_event_labels(video_id, n_frames, label_width=5):
    """Create binary frame-level event labels.

    Labels a window of `label_width` frames centered on each event start as positive.
    This gives the model some tolerance for detecting events slightly early or late.

    Args:
        video_id: EPIC-Kitchens video ID
        n_frames: number of frames (after frame_skip)
        label_width: number of frames around event start to mark positive

    Returns:
        labels: (n_frames,) float tensor, 1.0 = event, 0.0 = non-event
    """
    labels = torch.zeros(n_frames)
    events = load_video_labels(video_id)
    half = label_width // 2

    for event in events:
        # Convert absolute frame to feature index (accounting for frame_skip)
        feat_idx = event["start_frame"] // cfg.frame_skip

        for offset in range(-half, half + 1):
            idx = feat_idx + offset
            if 0 <= idx < n_frames:
                labels[idx] = 1.0

    return labels


class EventDataset(Dataset):
    """PyTorch dataset for training Mamba event detection.

    Loads pre-extracted CLIP features and creates fixed-length windows
    with binary event labels for training.
    """

    def __init__(self, video_ids, feature_dir="data/features", seq_len=None, stride=None):
        """
        Args:
            video_ids: list of video IDs to include
            feature_dir: directory containing pre-extracted .pt feature files
            seq_len: window length in frames (default: cfg.seq_len)
            stride: step between windows (default: seq_len // 2 for 50% overlap)
        """
        self.feature_dir = Path(feature_dir)
        self.seq_len = seq_len or cfg.seq_len
        self.stride = stride or self.seq_len // 2

        self.windows = []  # list of (video_id, start_idx)
        self.features = {}  # video_id → (N, d_model) tensor
        self.labels = {}  # video_id → (N,) tensor

        for video_id in video_ids:
            feat_path = self.feature_dir / f"{video_id}.pt"
            if not feat_path.exists():
                print(f"  Warning: features not found for {video_id}, skipping")
                continue

            feats = torch.load(feat_path, weights_only=True).float()
            n_frames = feats.shape[0]

            if n_frames < self.seq_len:
                continue

            self.features[video_id] = feats
            self.labels[video_id] = make_event_labels(video_id, n_frames)

            # Create overlapping windows
            for start in range(0, n_frames - self.seq_len + 1, self.stride):
                self.windows.append((video_id, start))

        # Compute positive weight for balanced BCE
        total_pos = sum(l.sum().item() for l in self.labels.values())
        total_neg = sum(l.numel() - l.sum().item() for l in self.labels.values())
        self.pos_weight = total_neg / max(total_pos, 1.0)

        print(f"  Dataset: {len(self.windows)} windows from {len(self.features)} videos")
        print(f"  Positive frames: {int(total_pos)}, Negative: {int(total_neg)}, "
              f"pos_weight: {self.pos_weight:.1f}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        video_id, start = self.windows[idx]
        end = start + self.seq_len

        features = self.features[video_id][start:end]  # (seq_len, d_model)
        labels = self.labels[video_id][start:end]  # (seq_len,)

        return features, labels