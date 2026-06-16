import collections
import numpy as np


class EventGate:
    def __init__(self, cfg):
        self.window_size = cfg.window_size
        self.cooldown = cfg.cooldown
        self.confirm_frames = cfg.confirm_frames
        self.k_base = cfg.k

        self.history = collections.deque(maxlen=self.window_size)
        self.last_event_frame = -self.cooldown
        self.candidate_frame = None
        self.confirm_count = 0
        self.mean_at_spike = 0.0  # snapshot, not moving mean

    def check_event(self, features, frame_idx):
        score = features["event_score"]
        self.history.append(score)

        # warm-up: gate only fires once history is full
        if len(self.history) < self.window_size:
            return False

        mean = np.mean(self.history)
        std = np.std(self.history) + 1e-6
        threshold = mean + self.k_base * std

        # stage 2: candidate is alive, check sustained signal
        if self.candidate_frame is not None:
            if score > self.mean_at_spike:
                self.confirm_count += 1
                if self.confirm_count >= self.confirm_frames:
                    self.last_event_frame = self.candidate_frame
                    trigger_frame = self.candidate_frame
                    self.candidate_frame = None
                    self.confirm_count = 0
                    return trigger_frame
            else:
                # signal dropped, reject candidate
                self.candidate_frame = None
                self.confirm_count = 0

        # stage 1: detect new spike
        if (
            score > threshold
            and self.candidate_frame is None
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.candidate_frame = frame_idx
            self.mean_at_spike = mean
            self.confirm_count = 0

        return False
