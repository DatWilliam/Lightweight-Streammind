import collections
import numpy as np

class EventGate:
    def __init__(self, cfg):
        self.window_size = cfg.window_size
        self.cooldown = cfg.cooldown
        self.k = cfg.k
        self.history = collections.deque(maxlen=self.window_size) # sliding window buffer
        self.last_event_frame = -self.cooldown

    def check_event(self, features, frame_idx):
        score = features["event_score"]
        self.history.append(score)

        if len(self.history) < self.window_size:
            return False

        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6 # standard deviation
        threshold = mean + self.k * std

        if (
            score > threshold
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.last_event_frame = frame_idx
            return frame_idx

        return False