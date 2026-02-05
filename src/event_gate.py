import collections
import numpy as np

class EventGate:
    def __init__(
        self,
        window_size=60,   # amount of frames used for context
        k=1.5,            # sensitivity
        cooldown=45       # between two events
    ):
        self.window_size = window_size
        self.k = k
        self.cooldown = cooldown

        self.history = collections.deque(maxlen=window_size) # sliding window buffer
        self.last_event_frame = -cooldown

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
