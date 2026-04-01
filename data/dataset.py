import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class EPFEDataset(Dataset):
    """
    Lädt gecachte CLIP-Features + generiert Frame-Level Labels aus Event-Annotationen.
    Gibt nicht-überlappende Fenster der Länge buffer_size zurück.

    Labels: 1.0 für alle Frames innerhalb von ±event_radius um einen Event-Start, sonst 0.0
    """

    def __init__(self, cfg, dataset: str, split: str, event_radius: int = 15):
        if dataset == "epickitchen":
            from data.prepare_epickitchen import load_video_labels
        else:
            from data.prepare_soccernet import load_video_labels

        self.buffer_size = cfg.mamba_buffer_size
        self.sequences = []  # Liste von (features (buffer_size, 512), labels (buffer_size,))

        video_ids = getattr(cfg, f"video_ids_{split}")

        for video_id in video_ids:
            cache_path = cfg.DATA_DIR / dataset / str(Path(video_id).with_suffix(".npz"))
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Cache nicht gefunden: {cache_path}\n"
                    f"Erst ausführen: python -m utils.build_cache {dataset}"
                )

            data = np.load(str(cache_path))
            features = data["features"]  # (F, dim) float32

            # Frame-Level Labels
            gt_events = load_video_labels(video_id)
            event_starts = set(e["start_frame"] for e in gt_events)

            labels = np.zeros(len(features), dtype=np.float32)
            for start_frame in event_starts:
                for offset in range(-event_radius, event_radius + 1):
                    target = start_frame + offset
                    # frame_idx starts at 1, array is 0-indexed
                    idx = target - 1
                    if 0 <= idx < len(features):
                        labels[idx] = 1.0

            # Nicht-überlappende Fenster der Länge buffer_size
            F = len(features)
            for i in range(0, F - self.buffer_size + 1, self.buffer_size):
                self.sequences.append((
                    features[i:i + self.buffer_size].copy(),
                    labels[i:i + self.buffer_size].copy(),
                ))

        # pos_weight für BCE: #negativ / #positiv
        all_labels = np.concatenate([s[1] for s in self.sequences])
        n_pos = all_labels.sum()
        n_neg = len(all_labels) - n_pos
        self.pos_weight = float(n_neg / max(n_pos, 1))
        print(f"Dataset [{split}]: {len(self.sequences)} Sequenzen | "
              f"pos={int(n_pos)} neg={int(n_neg)} | pos_weight={self.pos_weight:.1f}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        features, labels = self.sequences[idx]
        return torch.from_numpy(features), torch.from_numpy(labels)