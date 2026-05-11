import torch
import torch.nn as nn
import numpy as np
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    # pre-norm Mamba block + residual
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return self.mamba(self.norm(x)) + x


class EPFECached(nn.Module):
    """Mamba-based EPFE for the cached pipeline (CLIP features pre-extracted)."""

    def __init__(self, cfg, use_mamba: bool = True):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_mamba = use_mamba

        feature_dim = 512
        self.mamba = nn.Sequential(MambaLayer(feature_dim), nn.LayerNorm(feature_dim))
        self.score_head = nn.Linear(feature_dim, 1)
        self.buffer_size = cfg.mamba_buffer_size

        self.to(self.device)

    def forward_train(self, clip_features):
        # clip_features: (batch, seq_len, 512). Returns (batch, seq_len) scores.
        if self.use_mamba:
            assert clip_features.shape[1] == self.buffer_size, \
                f"seq_len {clip_features.shape[1]} != buffer_size {self.buffer_size}"
            out = self.mamba(clip_features)
        else:
            out = clip_features  # no temporal context
        return self.score_head(out).squeeze(-1)

    @torch.no_grad()
    def score_video(self, features_np, batch_size: int = 256):
        """Bulk inference for one video. Same outputs as a per-frame loop, GPU-batched."""
        N = len(features_np)
        feats = torch.from_numpy(features_np).to(self.device)

        if not self.use_mamba:
            return self.score_head(feats).squeeze(-1).cpu().numpy()

        if N < self.buffer_size:
            return np.zeros(N, dtype=np.float32)

        # sliding windows: (N - buffer_size + 1, buffer_size, D)
        windows = feats.unfold(0, self.buffer_size, 1).permute(0, 2, 1).contiguous()

        out_scores = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            mamba_out = self.mamba(batch)              # (B, buffer_size, D)
            last_token = mamba_out[:, -1]              # (B, D)
            out_scores.append(self.score_head(last_token).squeeze(-1))

        scores_concat = torch.cat(out_scores, dim=0)   # (N - buffer_size + 1,)

        # first (buffer_size - 1) frames have no full context: pad with 0
        full_scores = torch.zeros(N, device=self.device)
        full_scores[self.buffer_size - 1:] = scores_concat
        return full_scores.cpu().numpy()

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
