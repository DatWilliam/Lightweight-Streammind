from dataclasses import dataclass

@dataclass
class Config:
    epickitchen_video_root: str = "../data/epickitchen/P01"
    ann_dir: str = "../data/epickitchen/annotations"
    clip_model: str = "ViT-B/32" #ViT-L/14@336px

    fps: int = 50 # video fps
    frame_skip: int = 2 # for performance

    alpha: float = 0.85 # EMA

    window_size: int = 150 # how many frames to keep for context
    k: float = 2.0 # sensitivity
    cooldown: int = 40 # min frames between events
    confirm_frames: int = 3 # consecutive frames above mean to confirm event

    max_tolerance: int = 75 # cap for adaptive tolerance (1.5s at 50fps)
    tolerance_fraction: float = 0.5 # fraction of event duration used as tolerance


cfg = Config()