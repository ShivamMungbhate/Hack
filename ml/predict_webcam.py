'''
import json
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "signbridge_svm.joblib"
LABELS_PATH = ROOT / "models" / "sign_labels.json"
LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.30
MAX_MISSED_FRAMES = 5


def extract_features(result):
    """Extract 126 wrist-relative features using consistent hand slots."""
    features = np.zeros(126, dtype=np.float32)

    for hand, handedness in zip(
        result.hand_landmarks[:2],
        result.handedness[:2],
    ):
        hand_name = handedness[0].category_name
        hand_index = 0 if hand_name == "Left" else 1
        start = hand_index * 63

        wrist = hand[0]
        points = np.array(
            [
                [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]
                for p in hand
            ],
            dtype=np.float32,
        )

        scale = np.linalg.norm(points[9])
        if scale > 1e-6:
            points /= scale

        features[start:start + 63] = points.flatten()

    return features


def main():
    for path in (MODEL_PATH, LABELS_PATH, LANDMARKER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model = joblib.load(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    print("Loaded labels:", labels)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Check camera permissions.")
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    last_timestamp_ms = -1
    missed_frames = 0
    prediction = "Show one sign"
    confidence = 0.0

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = max(
                    int(time.monotonic() * 1000),
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    missed_frames = 0
                    sequence.append(extract_features(result))
                else:
                    missed_frames += 1
                    sequence.append(np.zeros(126, dtype=np.float32))

                    if missed_frames > MAX_MISSED_FRAMES:
                        sequence.clear()
                        prediction = "Show hand clearly"
                        confidence = 0.0

                if (
                    len(sequence) == SEQUENCE_LENGTH
                    and missed_frames == 0
                ):
                    X = np.stack(sequence)

                    model_features = np.concatenate(
                        [
                            X.mean(axis=0),
                            X.std(axis=0),
                            X[-1] - X[0],
                        ]
                    ).reshape(1, -1)

                    probabilities = model.predict_proba(model_features)[0]
                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])
                    best_label = str(model.classes_[best_index])

                    print(
                        f"Prediction: {best_label} | "
                        f"Score: {confidence:.3f}"
                    )

                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction = best_label
                    else:
                        prediction = "Uncertain"

                if missed_frames > 0:
                    display_prediction = "Show hand clearly"
                else:
                    display_prediction = prediction

                cv2.putText(
                    frame,
                    f"Sign: {display_prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                if len(sequence) < SEQUENCE_LENGTH:
                    status = (
                        f"Collecting frames: "
                        f"{len(sequence)}/{SEQUENCE_LENGTH}"
                    )
                else:
                    status = f"Model score: {confidence:.0%}"

                cv2.putText(
                    frame,
                    status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Q = quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("SignBridge - ISL Prototype", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''


import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

# Allow importing the agent package from the project root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.adapter import AgentAdapter


MODEL_PATH = ROOT / "models" / "signbridge_svm.joblib"
LABELS_PATH = ROOT / "models" / "sign_labels.json"
LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.30
MAX_MISSED_FRAMES = 5
FEATURE_COUNT = 126


def extract_features(result):
    """Extract 126 wrist-relative features using consistent hand slots."""
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)

    for hand, handedness in zip(
        result.hand_landmarks[:2],
        result.handedness[:2],
    ):
        hand_name = handedness[0].category_name
        hand_index = 0 if hand_name == "Left" else 1
        start = hand_index * 63

        wrist = hand[0]
        points = np.array(
            [
                [p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]
                for p in hand
            ],
            dtype=np.float32,
        )

        scale = np.linalg.norm(points[9])
        if scale > 1e-6:
            points /= scale

        features[start:start + 63] = points.flatten()

    return features


def make_prediction_event(sign, confidence):
    """Build an event accepted by the SignBridge agent."""
    return {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": int(time.time() * 1000),
        "sign": str(sign),
        "confidence": float(confidence),
        "alternatives": [],
        "gloss_buffer": [],
        "sentence": None,
        "is_final": False,
    }


def main():
    for path in (MODEL_PATH, LABELS_PATH, LANDMARKER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model = joblib.load(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    print("Loaded labels:", labels)

    # Agent runs in this process; no second API server is required.
    agent_adapter = AgentAdapter()
    last_draft = None

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open webcam. Check camera permissions.")
        return

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    last_timestamp_ms = -1
    missed_frames = 0
    prediction = "Show one sign"
    confidence = 0.0

    # Send an agent event only when the predicted label changes.
    # The sequence manager also handles stabilization and duplicates.
    last_sent_sign = None

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Could not read a webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                timestamp_ms = max(
                    int(time.monotonic() * 1000),
                    last_timestamp_ms + 1,
                )
                last_timestamp_ms = timestamp_ms

                result = landmarker.detect_for_video(image, timestamp_ms)

                if result.hand_landmarks:
                    missed_frames = 0
                    sequence.append(extract_features(result))
                else:
                    missed_frames += 1
                    sequence.append(
                        np.zeros(FEATURE_COUNT, dtype=np.float32)
                    )

                    if missed_frames > MAX_MISSED_FRAMES:
                        sequence.clear()
                        prediction = "Show hand clearly"
                        confidence = 0.0

                        if last_sent_sign is not None:
                            agent_adapter.reset_gesture_lock()
                            last_sent_sign = None

                if (
                    len(sequence) == SEQUENCE_LENGTH
                    and missed_frames == 0
                ):
                    X = np.stack(sequence)

                    model_features = np.concatenate(
                        [
                            X.mean(axis=0),
                            X.std(axis=0),
                            X[-1] - X[0],
                        ]
                    ).reshape(1, -1)

                    probabilities = model.predict_proba(
                        model_features
                    )[0]

                    best_index = int(np.argmax(probabilities))
                    confidence = float(probabilities[best_index])
                    best_label = str(model.classes_[best_index])

                    prediction = (
                        best_label
                        if confidence >= CONFIDENCE_THRESHOLD
                        else "Uncertain"
                    )

                    print(
                        f"Prediction: {prediction} | "
                        f"Score: {confidence:.3f}"
                    )

                    # Ignore COLLECTING-like states and low-confidence
                    # predictions. Process a label once until it changes.
                    if (
                        prediction not in ("Uncertain", "COLLECTING")
                        and prediction != last_sent_sign
                    ):
                        event = make_prediction_event(
                            prediction,
                            confidence,
                        )
                        agent_result = agent_adapter.process_event(event)
                        last_sent_sign = prediction

                        if agent_result["sequence"].get("accepted"):
                            draft = agent_result.get("message") or {}
                            sentence = draft.get("sentence")

                            if sentence and sentence != last_draft:
                                print(f"Agent draft: {sentence}")
                                last_draft = sentence

                    elif prediction == "Uncertain":
                        # Permit a later clear prediction to be processed.
                        last_sent_sign = None

                display_prediction = (
                    "Show hand clearly"
                    if missed_frames > 0
                    else prediction
                )

                cv2.putText(
                    frame,
                    f"Sign: {display_prediction}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                if len(sequence) < SEQUENCE_LENGTH:
                    status = (
                        f"Collecting frames: "
                        f"{len(sequence)}/{SEQUENCE_LENGTH}"
                    )
                else:
                    status = f"Model score: {confidence:.0%}"

                cv2.putText(
                    frame,
                    status,
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Q = quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("SignBridge - ISL Prototype", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
