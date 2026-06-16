import clip
from PIL import Image
import cv2
import torch
import numpy as np


class EPFE:
    """Live EMA-EPFE: CLIP per frame + L2-distance to EMA-state. Parameter-free."""

    def __init__(self, cfg):
        self.state = None
        self.alpha = cfg.alpha

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.model.eval()

        for param in self.model.parameters(): # freeze
            param.requires_grad = False

    def process_batch(self, frames):
        images = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in frames
        ]).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()

        results = []
        for feature in features:
            if self.state is None:
                self.state = feature.copy()
                results.append({"event_score": 0.0})
            else:
                event_score = float(np.linalg.norm(feature - self.state))
                self.state = self.alpha * feature + (1 - self.alpha) * self.state
                results.append({"event_score": event_score})
        return results

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
            self.state = feature.copy()
            return {"event_score": 0.0}

        delta = feature - self.state
        event_score = float(np.linalg.norm(delta))
        self.state = self.alpha * feature + (1 - self.alpha) * self.state  # EMA

        return {
            "event_score": float(event_score)
        }
