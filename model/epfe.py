import clip
from PIL import Image
import cv2
import torch
import numpy as np
from config import cfg

class EPFE:
    def __init__(self):
        self.state = None
        self.alpha = cfg.alpha

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.model.eval()

        for param in self.model.parameters(): # freeze
            param.requires_grad = False

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        image = Image.fromarray(frame) # convert np_arr to PIL image

        # open ai implementation
        image = (self.preprocess(image).unsqueeze(0).to(self.device))
        with torch.no_grad():
            feature = self.model.encode_image(image) # image to vector
            feature = feature / feature.norm(dim=-1, keepdim=True) # normalisation
            feature = feature.squeeze(0).cpu().numpy()

        if self.state is None:
            self.state = feature
            return {"event_score": 0.0}

        delta = feature -self.state
        event_score = np.linalg.norm(delta)

        self.state = self.alpha * self.state + (1 - self.alpha) * feature # EMA

        return {
            "event_score": float(event_score),
            "delta": delta,
            "state_norm": float(np.linalg.norm(self.state))
        }

