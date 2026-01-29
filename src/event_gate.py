# aka Cognition-Gate: evaluates the current event and decides if the llm should be triggered

class EventGate:
    # Thresholds : exceeding triggers event
    def __init__(self, motion_thresh=0.01, scene_thresh=0.01):
        self.motion_thresh = motion_thresh
        self.scene_thresh = scene_thresh
        self.last_trigger_frame = -1  # how many frames since last event trigger

    # Called for every frame, checks if event trigger
    def check_event(self, features, current_frame_idx):
        motion = features["motion_score"]
        scene = features["scene_change_score"]

        if (motion > self.motion_thresh or scene > self.scene_thresh) and \
           (current_frame_idx - self.last_trigger_frame > 5):
            self.last_trigger_frame = current_frame_idx
            return True
        return False
