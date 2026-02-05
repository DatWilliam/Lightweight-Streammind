import clip
from PIL import Image
import cv2
import torch
import numpy as np

class Perception:
    def __init__(self, alpha=0.7):
        self.state = None
        self.alpha = alpha

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess = clip.load(
            #"ViT-L/14@336px", # same version as streammind
            "ViT-B/32",
            device=self.device
        )
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
            return {"event_score": 0.0} # no comparison

        # ema
        new_state = self.alpha * self.state + (1 - self.alpha) * feature

        # Event score = semantic change
        event_score = np.linalg.norm(new_state - self.state)

        self.state = new_state

        return {
            "event_score": float(event_score),
            "state_norm": float(np.linalg.norm(self.state))
        }

