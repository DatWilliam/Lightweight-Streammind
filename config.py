from dataclasses import dataclass
from pathlib import Path

@dataclass
class BaseConfig:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    clip_model: str = "ViT-B/32"# "ViT-L/14@336px" -> takes > 100h
    window_size: int = 30  # context frames
    mamba_buffer_size: int = 16  # sliding window context for Mamba

class SoccerNetConfig(BaseConfig):
    fps: int = 25
    k: float = 3.5
    k_min: float = 2.5
    k_max: float = 4
    cooldown: int = 50
    confirm_frames: int = 10
    tolerance: int = 5  # seconds

    # video_id = relativer Pfad ab data/soccernet/ (ohne fuehrendes /)
    # Test
    video_ids_test = [
        "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv",
        "england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/2_224p.mkv",
        "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/1_224p.mkv",
        "england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea/2_224p.mkv",
        "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2016-08-20 - 17-00 Burnley 2 - 0 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-08-20 - 17-00 Burnley 2 - 0 Liverpool/2_224p.mkv",
    ]

    # Train
    video_ids_train = [
        # 2014-2015
        
        "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal/1_224p.mkv",
        "england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal/2_224p.mkv",
        "england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/1_224p.mkv",
        "england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United/2_224p.mkv",
        "england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool/1_224p.mkv",
        "england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool/2_224p.mkv",
        # 2015-2016
        "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace/1_224p.mkv",
        "england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace/2_224p.mkv",
        "england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford/1_224p.mkv",
        "england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford/2_224p.mkv",
        "england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea/1_224p.mkv",
        "england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea/2_224p.mkv",
        "england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City/1_224p.mkv",
        "england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City/2_224p.mkv",
        "england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham/1_224p.mkv",
        "england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham/2_224p.mkv",
        "england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa/1_224p.mkv",
        "england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa/2_224p.mkv",
        "england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa/1_224p.mkv",
        "england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa/2_224p.mkv",
        "england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool/1_224p.mkv",
        "england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool/2_224p.mkv",
        "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom/1_224p.mkv",
        "england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom/2_224p.mkv",
        "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool/1_224p.mkv",
        "england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool/2_224p.mkv",
        "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea/1_224p.mkv",
        "england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea/2_224p.mkv",
        "england_epl/2015-2016/2015-12-05 - 20-30 Chelsea 0 - 1 Bournemouth/1_224p.mkv",
        "england_epl/2015-2016/2015-12-05 - 20-30 Chelsea 0 - 1 Bournemouth/2_224p.mkv",
        "england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland/1_224p.mkv",
        "england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland/2_224p.mkv",
        "england_epl/2015-2016/2015-12-26 - 18-00 Manchester City 4 - 1 Sunderland/1_224p.mkv",
        "england_epl/2015-2016/2015-12-26 - 18-00 Manchester City 4 - 1 Sunderland/2_224p.mkv",
        "england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea/1_224p.mkv",
        "england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea/2_224p.mkv",
        "england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom/1_224p.mkv",
        "england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom/2_224p.mkv",
        "england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United/1_224p.mkv",
        "england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United/2_224p.mkv",
        "england_epl/2015-2016/2016-02-14 - 19-15 Manchester City 1 - 2 Tottenham/1_224p.mkv",
        "england_epl/2015-2016/2016-02-14 - 19-15 Manchester City 1 - 2 Tottenham/2_224p.mkv",
        "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City/1_224p.mkv",
        "england_epl/2015-2016/2016-03-02 - 23-00 Liverpool 3 - 0 Manchester City/2_224p.mkv",
        "england_epl/2015-2016/2016-03-05 - 18-00 Chelsea 1 - 1 Stoke City/1_224p.mkv",
        "england_epl/2015-2016/2016-03-05 - 18-00 Chelsea 1 - 1 Stoke City/2_224p.mkv",
        "england_epl/2015-2016/2016-03-19 - 18-00 Chelsea 2 - 2 West Ham/1_224p.mkv",
        "england_epl/2015-2016/2016-03-19 - 18-00 Chelsea 2 - 2 West Ham/2_224p.mkv",
        "england_epl/2015-2016/2016-04-09 - 17-00 Swansea 1 - 0 Chelsea/1_224p.mkv",
        "england_epl/2015-2016/2016-04-09 - 17-00 Swansea 1 - 0 Chelsea/2_224p.mkv",
        "england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom/1_224p.mkv",
        "england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom/2_224p.mkv",
        "england_epl/2015-2016/2016-05-07 - 17-00 Sunderland 3 - 2 Chelsea/1_224p.mkv",
        "england_epl/2015-2016/2016-05-07 - 17-00 Sunderland 3 - 2 Chelsea/2_224p.mkv",
        # 2016-2017
        "england_epl/2016-2017/2016-08-20 - 19-30 Leicester 0 - 0 Arsenal/1_224p.mkv",
        "england_epl/2016-2017/2016-08-20 - 19-30 Leicester 0 - 0 Arsenal/2_224p.mkv",
        "england_epl/2016-2017/2016-09-10 - 17-00 Arsenal 2 - 1 Southampton/1_224p.mkv",
        "england_epl/2016-2017/2016-09-10 - 17-00 Arsenal 2 - 1 Southampton/2_224p.mkv",
        "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2016-09-17 - 17-00 Hull City 1 - 4 Arsenal/1_224p.mkv",
        "england_epl/2016-2017/2016-09-17 - 17-00 Hull City 1 - 4 Arsenal/2_224p.mkv",
        "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea/1_224p.mkv",
        "england_epl/2016-2017/2016-09-24 - 19-30 Arsenal 3 - 0 Chelsea/2_224p.mkv",
        "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United/1_224p.mkv",
        "england_epl/2016-2017/2016-10-17 - 22-00 Liverpool 0 - 0 Manchester United/2_224p.mkv",
        "england_epl/2016-2017/2016-10-22 - 19-30 Liverpool 2 - 1 West Brom/1_224p.mkv",
        "england_epl/2016-2017/2016-10-22 - 19-30 Liverpool 2 - 1 West Brom/2_224p.mkv",
        "england_epl/2016-2017/2016-10-29 - 14-30 Sunderland 1 - 4 Arsenal/1_224p.mkv",
        "england_epl/2016-2017/2016-10-29 - 14-30 Sunderland 1 - 4 Arsenal/2_224p.mkv",
        "england_epl/2016-2017/2016-10-29 - 17-00 Tottenham 1 - 1 Leicester/1_224p.mkv",
        "england_epl/2016-2017/2016-10-29 - 17-00 Tottenham 1 - 1 Leicester/2_224p.mkv",
        "england_epl/2016-2017/2016-11-06 - 17-15 Liverpool 6 - 1 Watford/1_224p.mkv",
        "england_epl/2016-2017/2016-11-06 - 17-15 Liverpool 6 - 1 Watford/2_224p.mkv",
        "england_epl/2016-2017/2016-11-06 - 19-30 Leicester 1 - 2 West Brom/1_224p.mkv",
        "england_epl/2016-2017/2016-11-06 - 19-30 Leicester 1 - 2 West Brom/2_224p.mkv",
        "england_epl/2016-2017/2016-11-19 - 18-00 Southampton 0 - 0 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-11-19 - 18-00 Southampton 0 - 0 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2016-11-26 - 18-00 Liverpool 2 - 0 Sunderland/1_224p.mkv",
        "england_epl/2016-2017/2016-11-26 - 18-00 Liverpool 2 - 0 Sunderland/2_224p.mkv",
        "england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City/1_224p.mkv",
        "england_epl/2016-2017/2016-12-10 - 20-30 Leicester 4 - 2 Manchester City/2_224p.mkv",
        "england_epl/2016-2017/2016-12-11 - 19-30 Liverpool 2 - 2 West Ham/1_224p.mkv",
        "england_epl/2016-2017/2016-12-11 - 19-30 Liverpool 2 - 2 West Ham/2_224p.mkv",
        "england_epl/2016-2017/2016-12-14 - 22-45 Middlesbrough 0 - 3 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-12-14 - 22-45 Middlesbrough 0 - 3 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2016-12-19 - 23-00 Everton 0 - 1 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2016-12-19 - 23-00 Everton 0 - 1 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2016-12-27 - 20-15 Liverpool 4 - 1 Stoke City/1_224p.mkv",
        "england_epl/2016-2017/2016-12-27 - 20-15 Liverpool 4 - 1 Stoke City/2_224p.mkv",
        "england_epl/2016-2017/2016-12-31 - 20-30 Liverpool 1 - 0 Manchester City/1_224p.mkv",
        "england_epl/2016-2017/2016-12-31 - 20-30 Liverpool 1 - 0 Manchester City/2_224p.mkv",
        "england_epl/2016-2017/2017-01-02 - 15-30 Middlesbrough 0 - 0 Leicester/1_224p.mkv",
        "england_epl/2016-2017/2017-01-02 - 15-30 Middlesbrough 0 - 0 Leicester/2_224p.mkv",
        "england_epl/2016-2017/2017-01-02 - 18-00 Sunderland 2 - 2 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2017-01-02 - 18-00 Sunderland 2 - 2 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea/1_224p.mkv",
        "england_epl/2016-2017/2017-01-14 - 20-30 Leicester 0 - 3 Chelsea/2_224p.mkv",
        "england_epl/2016-2017/2017-01-15 - 19-00 Manchester United 1 - 1 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2017-01-15 - 19-00 Manchester United 1 - 1 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2017-01-31 - 23-00 Liverpool 1 - 1 Chelsea/1_224p.mkv",
        "england_epl/2016-2017/2017-01-31 - 23-00 Liverpool 1 - 1 Chelsea/2_224p.mkv",
        "england_epl/2016-2017/2017-02-27 - 23-00 Leicester 3 - 1 Liverpool/1_224p.mkv",
        "england_epl/2016-2017/2017-02-27 - 23-00 Leicester 3 - 1 Liverpool/2_224p.mkv",
        "england_epl/2016-2017/2017-04-26 - 21-45 Arsenal 1 - 0 Leicester/1_224p.mkv",
        "england_epl/2016-2017/2017-04-26 - 21-45 Arsenal 1 - 0 Leicester/2_224p.mkv",
        "england_epl/2016-2017/2017-05-13 - 14-30 Manchester City 2 - 1 Leicester/1_224p.mkv",
        "england_epl/2016-2017/2017-05-13 - 14-30 Manchester City 2 - 1 Leicester/2_224p.mkv",
        
    ]

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
