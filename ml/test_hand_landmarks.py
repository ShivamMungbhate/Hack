
import cv2
import mediapipe as mp
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "isl-isolated-8words"
    / "hello"
    /  "hello__ISL500__00087__Hello__session101__clip021.mp4"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}. "
            "Choose an existing MP4 from data/isl-isolated-8words/hello."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    detected_frames = 0
    total_frames = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                total_frames += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    cap.get(cv2.CAP_PROP_POS_MSEC)
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                if result.hand_landmarks:
                    detected_frames += 1

                    for hand in result.hand_landmarks:
                        for landmark in hand:
                            x = int(landmark.x * frame.shape[1])
                            y = int(landmark.y * frame.shape[0])
                            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                cv2.imshow("SignBridge - Hand Landmarks", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if total_frames % 25 == 0:
                    print(
                        f"Processed {total_frames} frames; "
                        f"hands detected in {detected_frames} frames"
                    )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n--- Results ---")
    print("Video:", VIDEO_PATH.name)
    print("Total frames:", total_frames)
    print("Frames with detected hands:", detected_frames)
    if total_frames:
        print(
            f"Detection rate: "
            f"{detected_frames / total_frames * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
