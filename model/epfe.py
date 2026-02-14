import clip
from PIL import Image
import cv2
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from model.mamba import MambaModel


class EPFE(nn.Module):
    """
    Event Perception Feature Extractor.

    Pipeline: Frame → CLIP (frozen) → Mamba (learnable) → perception token → event score

    The perception token is a rich spatio-temporal representation (d_model dim)
    that captures what's temporally relevant. The event head maps it to a binary
    event detection score. The perception token itself is preserved for future
    use (e.g., query-relevance scoring against text embeddings).
    """

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Frozen CLIP encoder ---
        self.clip_model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # --- Learnable Mamba temporal processor ---
        self.mamba = MambaModel(
            d_model=cfg.d_model,
            n_layers=cfg.n_mamba_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand
        )

        # --- Event detection head ---
        self.event_head = nn.Sequential(
            nn.Linear(cfg.d_model, 1),
        )

        # Streaming state for inference
        self._states = None

    def extract_clip_features(self, frame):
        """Extract frozen CLIP features from a single BGR frame.

        Args:
            frame: np.ndarray (H, W, 3) BGR format from OpenCV

        Returns:
            feature: torch.Tensor (1, d_model) normalized CLIP feature
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.clip_model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True)

        return feature.float()  # (1, d_model)

    def forward(self, features, states=None):
        """Forward pass for training on pre-extracted feature sequences.

        Args:
            features: (B, L, d_model) - sequence of CLIP features
            states: list of per-layer state dicts, or None

        Returns:
            event_scores: (B, L) - per-frame event probabilities
            perception_tokens: (B, L, d_model) - rich spatio-temporal features
            new_states: list of per-layer state dicts
        """
        perception_tokens, new_states = self.mamba(features, states)
        event_logits = self.event_head(perception_tokens).squeeze(-1)  # (B, L)
        event_scores = torch.sigmoid(event_logits)

        return event_scores, perception_tokens, new_states

    def process_frame(self, frame):
        """Streaming inference: process a single frame and carry state.

        Args:
            frame: np.ndarray (H, W, 3) BGR

        Returns:
            dict with 'event_score' (float) and 'perception_token' (tensor)
        """
        feature = self.extract_clip_features(frame)  # (1, d_model)
        feature = feature.unsqueeze(1)  # (1, 1, d_model) - single timestep

        with torch.no_grad():
            event_scores, perception_tokens, self._states = self.forward(feature, self._states)

        return {
            "event_score": float(event_scores[0, 0]),
            "perception_token": perception_tokens[0, 0]  # (d_model,)
        }

    def reset_state(self):
        """Reset streaming state for a new video."""
        self._states = None

    def save_checkpoint(self, path):
        """Save only the learnable parameters (Mamba + event head)."""
        torch.save({
            'mamba': self.mamba.state_dict(),
            'event_head': self.event_head.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """Load learnable parameters."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.mamba.load_state_dict(checkpoint['mamba'])
        self.event_head.load_state_dict(checkpoint['event_head'])