import numpy as np
import torch
from torch.utils.data import Dataset


class EPFEDataset(Dataset):
    """
    Cached-feature dataset for EPFE training.

    Loads CLIP features from .npz cache, builds frame-level binary labels
    (event_radius window around each GT event), and slices into
    non-overlapping sequences of buffer_size frames.
    """

    def __init__(self, cfg, dataset: str, split: str, event_radius: int = None):
        if event_radius is None:
            event_radius = getattr(cfg, "event_radius", 15)

        if dataset == "ego4d":
            from data.prepare_ego4d import load_video_labels, get_cache_path
        else:
            from data.prepare_soccernet import load_video_labels, get_cache_path

        self.buffer_size = cfg.mamba_buffer_size
        self.sequences = []  # list of (features, labels) pairs of shape (buffer_size, ...)

        video_ids = getattr(cfg, f"video_ids_{split}")

        for video_id in video_ids:
            cache_path = get_cache_path(video_id)
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cache not found: {cache_path}\n"
                    f"Run first: python -m utils.build_cache {dataset}"
                )

            data = np.load(str(cache_path))
            features = data["features"]  # (F, dim) float32

            # frame-level labels: 1 within ±event_radius of any GT event
            gt_events = load_video_labels(video_id)
            event_starts = set(e["start_frame"] for e in gt_events)

            labels = np.zeros(len(features), dtype=np.float32)
            for start_frame in event_starts:
                for offset in range(-event_radius, event_radius + 1):
                    target = start_frame + offset
                    # frame_idx is 1-indexed, array is 0-indexed
                    idx = target - 1
                    if 0 <= idx < len(features):
                        labels[idx] = 1.0

            # non-overlapping windows of length buffer_size
            F = len(features)
            for i in range(0, F - self.buffer_size + 1, self.buffer_size):
                self.sequences.append((
                    features[i:i + self.buffer_size].copy(),
                    labels[i:i + self.buffer_size].copy(),
                ))

        # pos_weight = #negative / #positive (for BCE)
        all_labels = np.concatenate([s[1] for s in self.sequences])
        n_pos = all_labels.sum()
        n_neg = len(all_labels) - n_pos
        self.pos_weight = float(n_neg / max(n_pos, 1))
        print(f"Dataset [{split}]: {len(self.sequences)} sequences | "
              f"pos={int(n_pos)} neg={int(n_neg)} | pos_weight={self.pos_weight:.1f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, labels = self.sequences[idx]
        return torch.from_numpy(features), torch.from_numpy(labels)
