
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "isl-isolated-8words"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
SEQUENCE_LENGTH = 30


def frame_features(landmarker, frame, timestamp):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp)
    output = np.zeros(126, dtype=np.float32)

    if not result.hand_landmarks:
        return output, False

    for hand, handedness in zip(
        result.hand_landmarks[:2], result.handedness[:2]
    ):
        slot = 0 if handedness[0].category_name == "Left" else 1
        points = np.array(
            [[p.x-hand[0].x, p.y-hand[0].y, p.z-hand[0].z]
             for p in hand],
            dtype=np.float32,
        )
        scale = np.linalg.norm(points[9])
        if scale > 1e-6:
            points /= scale
        output[slot*63:(slot+1)*63] = points.flatten()

    return output, True


def summarize(sequence):
    return np.concatenate([
        sequence.mean(axis=0),
        sequence.std(axis=0),
        sequence[-1] - sequence[0],
    ])


def main():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    original_X = np.load(ROOT / "data/processed/sequences.npy")
    y = np.load(ROOT / "data/processed/labels.npy",
                allow_pickle=True).astype(str)

    detected_sequences = []
    timestamp = 0

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        for label_dir in sorted(DATA.iterdir()):
            if not label_dir.is_dir():
                continue

            for video in sorted(label_dir.glob("*.mp4")):
                cap = cv2.VideoCapture(str(video))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frames = []
                index = 0

                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        features, found = frame_features(
                            landmarker, frame,
                            timestamp + int(index * 1000 / fps)
                        )
                        if found:
                            frames.append(features)
                        index += 1
                finally:
                    cap.release()

                timestamp += int(index * 1000 / fps) + 1000

                if not frames:
                    detected_sequences.append(None)
                    continue

                positions = np.linspace(
                    0, len(frames)-1, SEQUENCE_LENGTH
                ).astype(int)
                detected_sequences.append(
                    np.stack([frames[i] for i in positions])
                )

    # Match the metadata/video ordering used by the saved arrays.
    # Verify this ordering before treating the comparison as definitive.
    metadata = json.loads(
        (ROOT / "data/processed/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    # This script's directory iteration is sorted, matching extraction.
    if len(detected_sequences) != len(y) or any(
        seq is None for seq in detected_sequences
    ):
        raise RuntimeError("Some clips had no detected hands.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42
    )

    for name, sequences in [
        ("Original sampling", original_X),
        ("Detected-frame sampling", np.stack(detected_sequences)),
    ]:
        features = np.stack([summarize(seq) for seq in sequences])
        predictions = cross_val_predict(model, features, y, cv=cv)
        print(f"{name}: accuracy={accuracy_score(y, predictions):.3f}")


if __name__ == "__main__":
    main()
