from dataclasses import dataclass

@dataclass
class Config:
    epickitchen_video_root: str = "../data/epickitchen/P01"
    ann_dir: str = "../data/epickitchen/annotations"
    clip_model: str = "ViT-B/32" #ViT-L/14@336px

    fps: int = 50 # video fps
    frame_skip: int = 2 # for performance

    # --- Mamba architecture ---
    d_model: int = 512        # matches CLIP output dim
    n_mamba_layers: int = 2   # number of Mamba blocks
    d_state: int = 16         # SSM state dimension
    d_conv: int = 4           # local conv kernel size
    mamba_expand: int = 2     # expansion factor (d_inner = d_model * expand)

    # --- Training ---
    seq_len: int = 512        # training window length in frames
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.01

    # --- Inference ---
    event_threshold: float = 0.5  # perception token → event if score > threshold
    cooldown: int = 40            # min frames between events

    # --- Eval ---
    max_tolerance: int = 75 # cap for adaptive tolerance (1.5s at 50fps)
    tolerance_fraction: float = 0.5 # fraction of event duration used as tolerance


cfg = Config()