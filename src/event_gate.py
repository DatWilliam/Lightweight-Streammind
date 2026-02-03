class EventGate:
    def __init__(self, threshold=0.12, cooldown=30):
        self.threshold = threshold
        self.last_event_frame = -cooldown
        self.cooldown = cooldown

    def check_event(self, features, frame_idx):
        if (
            features["event_score"] > self.threshold
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.last_event_frame = frame_idx
            return True
        return False