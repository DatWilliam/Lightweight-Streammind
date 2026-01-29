# Lightweight Event Preserving Feature Extractor : Generates Perception-Token
import cv2
import numpy as np

# For now the model just looks for motion and scene change.
# Need to finetune later.
# Spatial Motion Features (Local)
# Pretrained YOLO / MobileNet for Object tracking
# Audio?
# Measure EPFE vs no EPFE
# Cap FPS to video to make it live
class Perception:
    def __init__(self):
        self.prev_frame = None

    def process_frame(self, frame):
        # Converts the frame to shades of gray to detect motion easier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion Score
        if self.prev_frame is None:
            motion_score = 0
        else:
            diff = cv2.absdiff(gray, self.prev_frame) # pixel diff this vs. prev frame
            motion_score = np.mean(diff) / 255.0  # average

        # To look for scene change we observe the global frame structure not the exact position of every pixel
        # We check if there is a lot of global change : How often is every shade of grey apparent
        scene_change_score = 0
        if self.prev_frame is not None:
            hist_prev = cv2.calcHist([self.prev_frame], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
            # 1 -> identical; 0 -> full different
            scene_change_score = 1 - cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

        self.prev_frame = gray.copy()

        return {
            "motion_score": motion_score,
            "scene_change_score": scene_change_score
        }
