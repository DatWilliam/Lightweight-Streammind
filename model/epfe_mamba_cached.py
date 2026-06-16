import torch
import torch.nn as nn
import numpy as np
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    # pre-norm Mamba block with residual connection
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return self.mamba(self.norm(x)) + x


class EPFECached(nn.Module):
    # Cached Mamba-EPFE: takes precomputed CLIP features -> Mamba -> score head.
    # No CLIP forward here; features come from the .npz cache.
    def __init__(self, cfg, use_mamba: bool = True):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_mamba = use_mamba

        # trainable Mamba block + outer LayerNorm
        feature_dim = 512
        self.mamba = nn.Sequential(MambaLayer(feature_dim), nn.LayerNorm(feature_dim))

        # score head turns the perception token into a scalar event score
        self.score_head = nn.Linear(feature_dim, 1)

        # window length the Mamba block sees per forward
        self.buffer_size = cfg.mamba_buffer_size

        self.to(self.device)

    def forward_train(self, clip_features):
        # training: (batch, buffer_size, 512) -> (batch, buffer_size) scores with gradients
        if self.use_mamba:
            assert clip_features.shape[1] == self.buffer_size, \
                f"seq_len {clip_features.shape[1]} != buffer_size {self.buffer_size}"
            out = self.mamba(clip_features)
        else:
            # ablation: no Mamba -> score head sees raw CLIP features (no temporal context)
            out = clip_features
        return self.score_head(out).squeeze(-1)

    @torch.no_grad()
    def score_video(self, features_np, batch_size: int = 256):
        # bulk inference for a single video, GPU-batched
        N = len(features_np)
        feats = torch.from_numpy(features_np).to(self.device)

        # ablation path: just per-frame score head
        if not self.use_mamba:
            return self.score_head(feats).squeeze(-1).cpu().numpy()

        # warm-up: too few frames -> no scores at all
        if N < self.buffer_size:
            return np.zeros(N, dtype=np.float32)

        # build sliding windows of length buffer_size: (N - buffer_size + 1, buffer_size, D)
        windows = feats.unfold(0, self.buffer_size, 1).permute(0, 2, 1).contiguous()

        # batch the windows through Mamba, keep only the last token of each
        out_scores = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            mamba_out = self.mamba(batch)              # (B, buffer_size, D)
            last_token = mamba_out[:, -1]              # (B, D)
            out_scores.append(self.score_head(last_token).squeeze(-1))

        scores_concat = torch.cat(out_scores, dim=0)   # (N - buffer_size + 1,)

        # the first (buffer_size - 1) frames have no full context -> pad with 0
        full_scores = torch.zeros(N, device=self.device)
        full_scores[self.buffer_size - 1:] = scores_concat
        return full_scores.cpu().numpy()

    def load_weights(self, path):
        # load trained Mamba + score-head weights from a .pt checkpoint
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.eval()
