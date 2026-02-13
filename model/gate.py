import collections
import numpy as np
from config import cfg

class EventGate:
    def __init__(self):
        self.window_size = cfg.window_size  # amount of frames used for context
        self.k = cfg.k                      # sensitivity
        self.cooldown = cfg.cooldown        # between two events

        self.history = collections.deque(maxlen=self.window_size) # sliding window buffer
        self.last_event_frame = -self.cooldown

    def check_event(self, features, frame_idx):
        score = features["event_score"]
        self.history.append(score)

        # fill the buffer first
        if len(self.history) < self.window_size:
            return False

        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6  # make sure standard deviation is never 0
        threshold = mean + self.k * std

        if (
            score > threshold
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.last_event_frame = frame_idx
            return True

        return False

"""
class EventGate:
    def __init__(self):
        self.window_size = cfg.window_size
        self.k = cfg.k
        self.cooldown = cfg.cooldown
        self.history = collections.deque(maxlen=cfg.window_size)
        self.last_event_frame = cfg.cooldown

        # NEU: Memory für Perception Tokens
        self.perception_memory = []  # Store wichtige Tokens

    def check_event(self, features, frame_idx):
        score = features["event_score"]
        self.history.append(score)

        # NEU: Speichere alle Perception Tokens
        self.perception_memory.append({
            "frame_idx": frame_idx,
            "features": features,
            "score": score
        })

        if len(self.history) < self.window_size:
            return False

        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6
        threshold = mean + self.k * std

        if (score > threshold and
                frame_idx - self.last_event_frame >= self.cooldown):
            self.last_event_frame = frame_idx

            # NEU: Gib relevante Tokens zurück
            return True, self._get_context_tokens(frame_idx)

        return False, None

    def _get_context_tokens(self, event_frame, context_window=30):
        context = [
            token for token in self.perception_memory
            if event_frame - context_window <= token["frame_idx"] <= event_frame
        ]
        return context
"""