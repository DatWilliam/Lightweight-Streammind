import clip
import torch
import torch.nn as nn
import cv2
from PIL import Image
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    # pre-norm Mamba block with residual connection
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return self.mamba(self.norm(x)) + x


class EPFE(nn.Module):
    # Live Mamba-EPFE: CLIP per frame -> rolling buffer -> Mamba -> score head
    def __init__(self, cfg):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CLIP image encoder, frozen
        self.clip_model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # trainable Mamba block on top of CLIP features
        feature_dim = 768 if "ViT-L" in cfg.clip_model else 512
        self.mamba = nn.Sequential(MambaLayer(feature_dim), nn.LayerNorm(feature_dim))

        # score head turns the perception token into a scalar event score
        self.score_head = nn.Linear(feature_dim, 1)

        # rolling buffer of the last buffer_size CLIP features (raw, not Mamba'd)
        self.buffer_size = cfg.mamba_buffer_size
        self.feature_buffer = []

        self.to(self.device)

    def _extract_clip_features(self, frames):
        # BGR frames -> normalized CLIP image embeddings (N, dim)
        images = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in frames
        ]).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        return features

    def _forward_mamba(self):
        # full buffer -> Mamba -> last token = current perception token -> event score
        seq = torch.stack(self.feature_buffer, dim=0).unsqueeze(0)  # (1, buffer_size, dim)
        out = self.mamba(seq)                                       # (1, buffer_size, dim)
        perception_token = out[0, -1]                               # (dim,)
        event_score = self.score_head(perception_token)             # (1,)
        return perception_token, event_score

    def process_frame(self, frame):
        # inference: one frame in, dict with event_score out (compatible with EventGate)
        feature = self._extract_clip_features([frame])[0]
        self.feature_buffer.append(feature.detach())

        # warm-up: until the buffer is full, no Mamba forward yet
        if len(self.feature_buffer) < self.buffer_size:
            return {"event_score": 0.0, "perception_token": None}

        with torch.no_grad():
            perception_token, event_score = self._forward_mamba()

        # drop oldest feature -> rolling window
        self.feature_buffer.pop(0)

        return {
            "event_score": float(event_score.item()),
            "perception_token": perception_token.detach()
        }

    def forward_train(self, clip_features):
        # training: takes precomputed CLIP features (batch, buffer_size, dim),
        # returns (batch, buffer_size) event scores with gradients
        assert clip_features.shape[1] == self.buffer_size, \
            f"seq_len {clip_features.shape[1]} != buffer_size {self.buffer_size}"
        out = self.mamba(clip_features)
        event_scores = self.score_head(out)
        return event_scores.squeeze(-1)

    def process_batch(self, frames):
        # inference: list of BGR frames -> list of dicts
        return [self.process_frame(f) for f in frames]
