import torch
import numpy as np


class EPFEEMACached:
    # Cached EMA-EPFE: takes precomputed CLIP features -> L2 distance to EMA state.
    # No CLIP forward, features come from the .npz cache.
    def __init__(self, cfg, use_mamba: bool = True):
        # use_mamba accepted for interface parity, but ignored (EMA has no Mamba)
        self.alpha = cfg.alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def score_video(self, features_np, batch_size: int = 256):
        # bulk inference for a single video; pure numpy, no GPU needed
        N = len(features_np)
        scores = np.zeros(N, dtype=np.float32)
        if N == 0:
            return scores

        # cache is already float32 -> no astype needed
        feats = np.ascontiguousarray(features_np, dtype=np.float32)

        # init EMA state with first frame; first score stays 0 (no reference yet)
        state = feats[0].copy()
        alpha = np.float32(self.alpha)
        one_minus_alpha = np.float32(1.0 - alpha)

        # for each frame: score = ||feat - state||, then update state
        for i in range(1, N):
            feat = feats[i]
            delta = feat - state
            # np.dot is faster than np.linalg.norm for small vectors
            scores[i] = np.sqrt(np.dot(delta, delta))
            state = alpha * feat + one_minus_alpha * state
        return scores

    def eval(self):
        # no-op, kept for interface parity
        pass

    def load_weights(self, path):
        # no learnable params, kept for interface parity
        pass
