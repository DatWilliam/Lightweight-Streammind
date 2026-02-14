import collections
import numpy as np
from config import cfg

class EventGate:
    def __init__(self):
        self.window_size = cfg.window_size
        self.cooldown = cfg.cooldown
        self.confirm_frames = cfg.confirm_frames
        self.history = collections.deque(maxlen=self.window_size) # sliding window buffer
        self.last_event_frame = -self.cooldown

        self.k_base = cfg.k
        self.k_min = cfg.k_min
        self.k_max = cfg.k_max

        # two-stage confirmation
        self.candidate_frame = None
        self.confirm_count = 0 # consecutive frames

    def check_event(self, features, frame_idx):
        score = features["event_score"]
        self.history.append(score)

        # fill the buffer first
        if len(self.history) < self.window_size:
            return False

        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6

        # adaptive k: scale k with coefficient of variation (CV)
        cv = std / (mean + 1e-6)
        k_adapted = np.clip(self.k_base * cv / 0.4, self.k_min, self.k_max)
        threshold = mean + k_adapted * std

        # stage 2: confirm event by checking sustained event-score
        if self.candidate_frame is not None:
            if score > mean:
                self.confirm_count += 1
                if self.confirm_count >= self.confirm_frames:
                    self.last_event_frame = self.candidate_frame
                    self.candidate_frame = None
                    self.confirm_count = 0
                    return True
            else:
                # score dropped below mean, reject candidate
                self.candidate_frame = None
                self.confirm_count = 0

        # stage 1: detect spike above threshold
        if (
            score > threshold
            and self.candidate_frame is None
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.candidate_frame = frame_idx
            self.confirm_count = 0

        return False