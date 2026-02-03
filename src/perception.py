# Lightweight Event Preserving Feature Extractor : Generates Perception-Token
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
        # openai/clip-vit-base-patch32 test?
        self.model, self.preprocess = clip.load("ViT-L/14@336px", device=self.device) # same version as streammind
        self.model.eval()

        for param in self.model.parameters(): # freeze
            param.requires_grad = False

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # https://github.com/openai/CLIP
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True) #L2-Normalising
            feature = feature.squeeze(0).cpu().numpy()

        if self.state is None:
            self.state = feature
            return {"event_score": 0.0}

        new_state = self.alpha * feature + (1 - self.alpha) * self.state

        # Event score = semantic change
        event_score = np.linalg.norm(new_state - self.state)

        self.state = new_state

        return {
            "event_score": float(event_score),
            "state_norm": float(np.linalg.norm(self.state))
        }

    def label_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        key_events = [
            "Kick-off",
            "Ball out of play",
            "Throw-in",
            "Corner",
            "Shots on target",
            "Offside",
            "Goal",
            "Clearance",
            "Foul",
            "Yellow card",
            "Red card",
            "Substitution"
        ]

        text = clip.tokenize(key_events).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            # normalisation
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity  = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices  = similarity[0].topk(3)

        results = []
        for value, index in zip(values, indices):
            results.append({
                "label": key_events[index],
                "confidence": f"{value.item():.1f}",
            })

        return results

