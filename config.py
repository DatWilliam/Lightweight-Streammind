from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"

    clip_model: str = "ViT-B/32" #ViT-L/14@336px

    fps: int = 50 # video fps (epickitchen)
    frame_skip: int = 3

    alpha: float = 0.85 # EMA

    window_size: int = 150 # how many frames to keep for context

    k: float = 2.0 # sensitivity
    k_min: float = 1.2  # floor for adaptive k
    k_max: float = 2.5  # ceiling for adaptive k

    cooldown: int = 40 # min frames between events
    confirm_frames: int = 3 # consecutive frames above mean to confirm event

    max_tolerance: int = 75 # cap for adaptive tolerance (1.5s at 50fps)
    tolerance_fraction: float = 0.5 # fraction of event duration used as tolerance

cfg = Config()