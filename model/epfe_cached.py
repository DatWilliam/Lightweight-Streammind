import torch
import torch.nn as nn
import numpy as np
from mamba_minimal import ResidualBlock, RMSNorm, ModelArgs


class EPFECached(nn.Module):
    """
    Mamba-based EPFE ohne CLIP — nutzt vorberechnete Features aus dem Cache.
    Identische Architektur wie EPFE, aber process_frame erwartet ein
    numpy-Array (512,) statt eines raw Frames.
    """

    def __init__(self, cfg):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        feature_dim = 512  # ViT-B/32; ViT-L/14 → 768
        mamba_args = ModelArgs(d_model=feature_dim, n_layer=1)
        self.mamba = nn.Sequential(ResidualBlock(mamba_args), RMSNorm(feature_dim))
        self.score_head = nn.Linear(feature_dim, 1)

        self.buffer_size = cfg.mamba_buffer_size
        self.feature_buffer = []

        self.to(self.device)

    def process_frame(self, feature):
        """
        feature: numpy array (512,) aus dem Cache
        Gibt dasselbe Dict zurück wie EPFE.process_frame.
        """
        if isinstance(feature, np.ndarray):
            feature = torch.from_numpy(feature).to(self.device)
        else:
            feature = feature.to(self.device)

        self.feature_buffer.append(feature)

        if len(self.feature_buffer) < self.buffer_size:
            return {"event_score": 0.0, "perception_token": None}

        with torch.no_grad():
            seq = torch.stack(self.feature_buffer, dim=0).unsqueeze(0)  # (1, buffer_size, 512)
            out = self.mamba(seq)                                         # (1, buffer_size, 512)
            perception_token = out[0, -1]                                 # (512,)
            event_score = self.score_head(perception_token)               # (1,)

        self.feature_buffer.pop(0)

        return {
            "event_score": float(event_score.item()),
            "perception_token": perception_token.detach()
        }

    def forward_train(self, clip_features):
        """
        Training-Modus: erwartet vorberechnete CLIP-Features.
        Args:
            clip_features: (batch, seq_len, 512) torch.Tensor
        Returns:
            event_scores: (batch, seq_len) torch.Tensor
        """
        assert clip_features.shape[1] == self.buffer_size, \
            f"seq_len {clip_features.shape[1]} != buffer_size {self.buffer_size}"
        out = self.mamba(clip_features)
        event_scores = self.score_head(out)
        return event_scores.squeeze(-1)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()