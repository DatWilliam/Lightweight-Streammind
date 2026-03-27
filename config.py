from dataclasses import dataclass
from pathlib import Path

@dataclass
class BaseConfig:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    clip_model: str = "ViT-B/32" # CLIP: ViT-L/14@336px
    window_size: int = 30  # context frames
    mamba_buffer_size: int = 16  # sliding window context for Mamba

class SoccerNetConfig(BaseConfig):
    fps: int = 25
    k: float = 3.5
    k_min: float = 2.5
    k_max: float = 4
    cooldown: int = 50
    confirm_frames: int = 10
    fixed_threshold: float = 0.2
    tolerance: int = 5  # seconds
    alpha: float = 0.26
    video_ids_train = ["3_224p.mkv", "4_224p.mkv", "5_224p.mkv", "6_224p.mkv", "7_224p.mkv", "8_224p.mkv", "9_224p.mkv", "10_224p.mkv", "11_224p.mkv", "12_224p.mkv", "13_224p.mkv", "14_224p.mkv"]
    video_ids_test = ["1_224p.mkv", "2_224p.mkv"]

class EpicKitchenConfig(BaseConfig):
    fps: int = 50
    k: float = 2
    k_min: float = 1.2
    k_max: float = 2.4
    cooldown: int = 25
    alpha: float = 0.05
    confirm_frames: int = 8
    tolerance: int = 1 # seconds
    fixed_threshold: float = 0.2
    video_ids_train = ["P01_102", "P01_103", "P01_104"]
    video_ids_test = [ "P01_106", "P07_111"]


def load_config(dataset: str = "soccernet"):
    configs = {
        "soccernet": SoccerNetConfig,
        "epickitchen": EpicKitchenConfig
    }
    return configs[dataset.lower()]()
