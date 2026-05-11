class EventGate:
    def __init__(self, cfg):
        self.cooldown = cfg.cooldown
        self.threshold = cfg.fixed_threshold
        self.last_event_frame = -self.cooldown

    def check_event(self, features, frame_idx):
        score = features["event_score"]

        if (
            score > self.threshold
            and frame_idx - self.last_event_frame >= self.cooldown
        ):
            self.last_event_frame = frame_idx
            return frame_idx

        return False