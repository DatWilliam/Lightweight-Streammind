import clip
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from mamba_minimal import ResidualBlock, RMSNorm, ModelArgs


class EPFE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── CLIP (eingefroren) ──────────────────────────────────────────
        self.clip_model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ── Mamba EPFE (trainierbar) ────────────────────────────────────
        feature_dim = 512  # ViT-B/32; ViT-L/14 → 768
        mamba_args = ModelArgs(d_model=feature_dim, n_layer=1)
        self.mamba = nn.Sequential(ResidualBlock(mamba_args), RMSNorm(feature_dim))

        # ── Score Head (trainierbar) ────────────────────────────────────
        # Lernt aus dem Perception Token einen event_score vorherzusagen
        self.score_head = nn.Linear(feature_dim, 1)

        # ── Buffer ─────────────────────────────────────────────────────
        self.buffer_size = cfg.mamba_buffer_size
        self.feature_buffer = []  # speichert rohe CLIP-Features (torch.Tensor)

        self.to(self.device)

    # ── CLIP Feature Extraction ─────────────────────────────────────────
    # CLIP bleibt immer eingefroren, daher no_grad hier okay
    def _extract_clip_features(self, frames):
        """frames: Liste von BGR numpy arrays → (N, 512) torch.Tensor"""
        images = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in frames
        ]).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        return features  # (N, 512), bleibt torch.Tensor

    # ── Core Forward ────────────────────────────────────────────────────
    def _forward_mamba(self):
        """
        Verarbeitet den aktuellen Buffer durch Mamba und score_head.
        Gibt perception_token und event_score zurück.
        Gradienten fließen durch Mamba und score_head.
        """
        # (1, buffer_size, 512)
        seq = torch.stack(self.feature_buffer, dim=0).unsqueeze(0)

        # Mamba forward — kein no_grad, Gradienten fließen
        out = self.mamba(seq)  # (1, buffer_size, 512)

        # Letzter Token ist der Perception Token des aktuellen Frames
        perception_token = out[0, -1]  # (512,)

        # Score Head: lernt event_score aus Perception Token
        event_score = self.score_head(perception_token)  # (1,)

        return perception_token, event_score

    # ── Inference: einzelner Frame ───────────────────────────────────────
    def process_frame(self, frame):
        """
        Inference-Modus: ein Frame, gibt event_score als float zurück.
        Schnittstelle bleibt kompatibel mit EventGate.
        """
        feature = self._extract_clip_features([frame])[0]  # (512,)
        self.feature_buffer.append(feature.detach())

        if len(self.feature_buffer) < self.buffer_size:
            return {"event_score": 0.0, "perception_token": None}

        with torch.no_grad():
            perception_token, event_score = self._forward_mamba()

        self.feature_buffer.pop(0)

        return {
            "event_score": float(event_score.item()),
            "perception_token": perception_token.detach()
        }

    # ── Training: Batch von Sequenzen ───────────────────────────────────
    def forward_train(self, clip_features):
        """
        Training-Modus: erwartet vorberechnete CLIP-Features.
        Gibt event_scores für alle Frames zurück.

        Args:
            clip_features: (batch, seq_len, 512) torch.Tensor

        Returns:
            event_scores: (batch, seq_len) torch.Tensor — Gradienten vorhanden
        """
        assert clip_features.shape[1] == self.buffer_size, \
            f"seq_len {clip_features.shape[1]} != buffer_size {self.buffer_size}"
        out = self.mamba(clip_features)          # (batch, seq_len, 512)
        event_scores = self.score_head(out)      # (batch, seq_len, 1)
        return event_scores.squeeze(-1)          # (batch, seq_len)

    # ── Inference: Batch von Frames ─────────────────────────────────────
    def process_batch(self, frames):
        """Inference-Modus: Liste von BGR frames → Liste von Dicts"""
        return [self.process_frame(f) for f in frames]