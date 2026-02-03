import json
from tqdm import tqdm
from typing import Any

from src.perception import Perception
from src.event_gate import EventGate
from src.utils import load_video

video_paths = [
    "./soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
]

test_data_path = "./soccernet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/Labels.json"

perception = Perception()
gate = EventGate()

SKIP_RATE = 3

frame_idx = 0
video_idx = 1

correct_matches = 0
false_positives = 0
false_negatives = 0
wrong_matches = 0

label_discovered = []

def get_label_for_frame() -> Any | None:
    with open(test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    tolerance = 15 # frames around event
    half = str(video_idx)


    for annotation in annotations:
        if str(annotation.get("half")) == half:
            annotation_frame = annotation.get("frame")
            if abs(annotation_frame - frame_idx) <= tolerance:
                if annotation_frame in label_discovered:
                    return None
                label_discovered.append(annotation_frame)
                return annotation.get("label")

    return None


for video_path in video_paths:

    for i, frame in tqdm(list(enumerate(load_video(video_path)))):
        frame_idx += 1
        if i % SKIP_RATE != 0:
            continue

        features = perception.process_frame(frame)

        # Cognition Gate (Silence/Response)
        if gate.check_event(features, frame_idx):

            predicted = perception.label_frame(frame)[0]['label']
            correct = get_label_for_frame()

            if predicted is not None and correct is None:
                false_positives += 1

            elif predicted is None and correct is not None:
                false_negatives += 1

            elif predicted == correct and predicted is not None:
                correct_matches += 1

            elif predicted != correct and predicted is not None and correct is not None:
                print(f" : wrong match: {predicted} != {correct}")
                wrong_matches += 1


    video_idx += 1

total = correct_matches + false_positives + false_negatives + wrong_matches
accuracy = correct_matches / total if total > 0 else 0

print("\n=== Evaluation Summary ===")
print(f"Total Evaluations: {total}")
print(f"Correct Matches:   {correct_matches}")
print(f"False Positives:   {false_positives}")
print(f"False Negatives:   {false_negatives}")
print(f"Wrong Matches:     {wrong_matches}")
print(f"Accuracy:          {accuracy:.2%}")










