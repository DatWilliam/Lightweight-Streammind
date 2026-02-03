import json
import copy
from pathlib import Path

FPS = 25

DATA_DIR = Path(__file__).resolve().parent

INPUT_PATH = DATA_DIR / "soccernet" / "england_epl" / "2014-2015" / \
    "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "Labels-v2.json"

OUTPUT_PATH = DATA_DIR / "soccernet" / "england_epl" / "2014-2015" / \
    "2015-02-21 - 18-00 Chelsea 1 - 1 Burnley" / "Labels.json"


def game_time_to_frame(game_time: str) -> int:
    _, time_part = game_time.split(" - ")
    minutes, seconds = map(int, time_part.split(":"))
    return (minutes * 60 + seconds) * FPS


def convert_labels(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = copy.deepcopy(data)

    # remove labels without visibility
    updated_annotations = [
        annotation for annotation in data['annotations']
        if annotation.get('visibility') != 'not shown'
    ]

    # add separate annotation for half
    for annotation in updated_annotations:
        if 'gameTime' in annotation:
            game_time = annotation['gameTime']
            half_value = game_time.split(' - ')[0].strip()
            annotation['half'] = half_value

    # add exact frame
    for annotation in updated_annotations:
        annotation["frame"] = game_time_to_frame(annotation["gameTime"])

    # remove unnecessary annotations
    for annotation in updated_annotations:
        del annotation["gameTime"]
        del annotation["team"]
        del annotation["visibility"]
        del annotation["position"]

    new_data['annotations'] = updated_annotations

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    convert_labels(INPUT_PATH, OUTPUT_PATH)
