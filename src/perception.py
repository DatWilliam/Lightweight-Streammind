# Lightweight Event Preserving Feature Extractor : Generates Perception-Token
import clip
from PIL import Image
import cv2
import torch
import numpy as np

class Perception:
    def __init__(self, alpha=0.15):
        self.state = None
        self.alpha = alpha

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # ViT-B/32 ViT-B/16 RN50x4 vs. frame skipping
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False #?

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # https://github.com/openai/CLIP
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(this.device)

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
            "a goal being scored",
            "players celebrating a goal",
            "penalty kick",
            "goalkeeper making a save",
            "referee showing a red card",
            "referee showing a yellow card",
            "players celebrating",
            "close-up replay",
            "wide view of football pitch",
            "ball in the penalty box"
        ]

        text = clip.tokenize(key_events).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            # normalisation
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity  = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices  = similarity [0].topk(3)

        results = []
        for value, index in zip(values, indices):
            results.append({
                "label": key_events[index],
                "confidence": f"{value.item():.1f}",
            })

        return results

