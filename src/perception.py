# Lightweight Event Preserving Feature Extractor : Generates Perception-Token

import cv2
import torch
import numpy as np
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class Perception:
    def __init__(self, alpha=0.05):
        self.prev_feature = None
        self.state = None # long term change
        self.alpha = alpha

        self.cnn = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
        self.cnn.classifier = torch.nn.Identity() # remove classifier, just feature
        self.cnn.eval() # to use model, not train

        for p in self.cnn.parameters():
            p.requires_grad = False # lock the parameters, no refining

    def process_frame(self, frame):
        frame = cv2.resize(frame, (224, 224)) # MobileNet takes 224x224
        frame = frame[:, :, ::-1] / 255.0 # BGR -> RGB
        # transform opencv to pytorch format
        frame = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()

        # no_grad since we dont train
        with torch.no_grad():
            feature = self.cnn(frame).squeeze().numpy()

        # for first frame, no prev
        if self.state is None:
            self.state = feature
            self.prev_feature = feature
            return {"event_score": 0.0}

        # Exponential Moving Average (!)
        new_state = self.alpha * feature + (1-self.alpha) * self.state

        # Event score = semantic change
        event_score = np.linalg.norm(new_state - self.state)

        self.state = new_state
        self.prev_feature = feature

        return {
            "event_score": float(event_score),
            "state_norm": float(np.linalg.norm(self.state))
        }

