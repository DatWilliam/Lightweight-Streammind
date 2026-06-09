import torch
import numpy as np


class EPFEEMACached:
    """
    EMA baseline on cached CLIP features. Parameter-free: score = ||feat - state||,
    state updated as state <- alpha * feat + (1 - alpha) * state.
    Exposes score_video / eval for the cached eval pipeline.
    """

    def __init__(self, cfg):
        self.alpha = cfg.alpha
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def score_video(self, features_np, batch_size: int = 256):
        N = len(features_np)
        scores = np.zeros(N, dtype=np.float32)
        if N == 0:
            return scores
        state = features_np[0].astype(np.float32).copy()
        for i in range(1, N):
            feat = features_np[i].astype(np.float32)
            scores[i] = float(np.linalg.norm(feat - state))
            state = self.alpha * feat + (1.0 - self.alpha) * state
        return scores

    def eval(self):
        # no-op; parameter-free, kept so the eval pipeline can call it
        pass


class EPFE:
    def __init__(self, cfg):
        self.state = None
        self.alpha = cfg.alpha

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        import clip  # lazy: only the live path needs CLIP/torchvision
        self.model, self.preprocess = clip.load(cfg.clip_model, device=self.device)
        self.model.eval()

        for param in self.model.parameters(): # freeze
            param.requires_grad = False

    def process_batch(self, frames):
        import cv2
        from PIL import Image
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
        import cv2
        from PIL import Image
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
