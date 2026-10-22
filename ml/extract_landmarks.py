
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "isl-isolated-8words"
OUTPUT_DIR = ROOT / "data" / "processed_low_threshold"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

# Fixed sequence length for the first prototype.
SEQUENCE_LENGTH = 30



def extract_frame_features(landmarker, frame, timestamp_ms):
    """Return 126 hand features using wrist-relative coordinates."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(image, timestamp_ms)
    features = np.zeros(126, dtype=np.float32)

    if not result.hand_landmarks:
        return features, False

    for hand, handedness in zip(
        result.hand_landmarks[:2],
        result.handedness[:2],
    ):
        hand_name = handedness[0].category_name

        # Always reserve slot 0 for Left and slot 1 for Right.
        hand_index = 0 if hand_name == "Left" else 1
        start = hand_index * 63
        # Use the wrist (landmark 0) as the origin.
        wrist = hand[0]
        points = np.array(
            [[p.x - wrist.x, p.y - wrist.y, p.z - wrist.z] for p in hand],
            dtype=np.float32,
        )

        # Scale by the distance from wrist to middle-finger MCP (landmark 9).
        # This reduces sensitivity to hand size and camera distance.
        scale = np.linalg.norm(points[9])
        if scale > 1e-6:
            points /= scale

        features[start:start + 63] = points.flatten()

    return features, True


def sample_video(landmarker, video_path, timestamp_offset_ms):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return None, 0, 0, timestamp_offset_ms

    frames = []
    detected = 0
    index = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Timestamp increases across the entire dataset.
            timestamp_ms = timestamp_offset_ms + int(index * 1000 / fps)

            features, has_hand = extract_frame_features(
                landmarker, frame, timestamp_ms
            )

            frames.append(features)
            detected += int(has_hand)
            index += 1
    finally:
        cap.release()

    if not frames:
        return None, 0, 0, timestamp_offset_ms

    positions = np.linspace(
        0, len(frames) - 1, SEQUENCE_LENGTH
    ).astype(int)

    sequence = np.stack([frames[i] for i in positions])

    # Add a one-second gap before the next clip.
    next_offset_ms = timestamp_offset_ms + int(len(frames) * 1000 / fps) + 1000

    return sequence, detected, len(frames), next_offset_ms


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = Options(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    samples = []
    timestamp_offset_ms = 0
    total_videos = 0
    skipped = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for label_dir in sorted(DATA_DIR.iterdir()):
            if not label_dir.is_dir():
                continue

            videos = sorted(label_dir.glob("*.mp4"))
            for video_path in videos:
                total_videos += 1
                sequence, detected, total_frames, timestamp_offset_ms = sample_video(
                landmarker, video_path, timestamp_offset_ms
                )

                if sequence is None:
                    skipped += 1
                    continue

                samples.append({
                    "label": label_dir.name.upper(),
                    "video": str(video_path.relative_to(ROOT)),
                    "sequence": sequence,
                    "detected_frames": detected,
                    "total_frames": total_frames,
                })

                print(
                    f"[{total_videos}] {label_dir.name}: "
                    f"{video_path.name} | "
                    f"hands in {detected}/{total_frames} frames"
                )

    if not samples:
        raise RuntimeError("No videos were successfully processed.")

    # Store arrays separately from metadata for easy model training.
    np.save(
        OUTPUT_DIR / "sequences.npy",
        np.stack([item["sequence"] for item in samples]),
    )
    np.save(
        OUTPUT_DIR / "labels.npy",
        np.array([item["label"] for item in samples]),
    )

    metadata = [
        {
            "label": item["label"],
            "video": item["video"],
            "detected_frames": item["detected_frames"],
            "total_frames": item["total_frames"],
        }
        for item in samples
    ]
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n--- Extraction summary ---")
    print("Videos discovered:", total_videos)
    print("Samples saved:", len(samples))
    print("Videos skipped:", skipped)
    print("Sequences shape:", (len(samples), SEQUENCE_LENGTH, 126))
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
