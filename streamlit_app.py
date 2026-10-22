
import json
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.adapter import AgentAdapter


MODEL_PATH = ROOT / "models" / "signbridge_svm.joblib"
LABELS_PATH = ROOT / "models" / "sign_labels.json"
LANDMARKER_PATH = ROOT / "models" / "hand_landmarker.task"

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.30
FEATURE_COUNT = 126
MAX_MISSED_FRAMES = 5


def extract_features(result):
    """Extract the same 126 wrist-relative features as the webcam script."""
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


@st.cache_resource
def load_model_assets():
    for path in (MODEL_PATH, LABELS_PATH, LANDMARKER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model = joblib.load(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels


def initialize_state():
    defaults = {
        "adapter": AgentAdapter(),
        "camera_on": False,
        "last_prediction": "Waiting for camera",
        "confidence": 0.0,
        "last_draft": "",
        "agent_status": "Waiting for a prediction",
        "agent_message": "",
        "agent_question": "",
        "agent_sequence": [],
        "history": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


st.set_page_config(
    page_title="SignBridge AI Agent",
    
    layout="wide",
)

initialize_state()

st.title(" SignBridge AI Agent")
st.caption("Live webcam sign recognition and message drafting")


try:
    model, labels = load_model_assets()
except Exception as exc:
    st.error(f"Could not load model assets: {exc}")
    st.stop()

with st.sidebar:
    st.header("Camera controls")

    camera_index = st.number_input(
        "Camera device index",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )

    st.caption(
        "Use camera index 0 for your default webcam. "
        "If another app is using the camera, close it first."
    )

    start_camera = st.button(
        "Start camera",
        type="primary",
        use_container_width=True,
    )

    stop_camera = st.button(
        "Stop camera",
        use_container_width=True,
    )

    if start_camera:
        st.session_state.camera_on = True

    if stop_camera:
        st.session_state.camera_on = False

    if st.button("Reset gesture lock", use_container_width=True):
        st.session_state.adapter.reset_gesture_lock()
        st.session_state.last_prediction = "Waiting for prediction"
        st.session_state.agent_status = "Gesture lock reset"

    if st.button("Clear sign sequence", use_container_width=True):
        st.session_state.adapter.clear()
        st.session_state.agent_sequence = []
        st.session_state.agent_message = ""
        st.session_state.agent_question = ""
        st.session_state.last_draft = ""
        st.session_state.agent_status = "Sequence cleared"


left, right = st.columns([1.25, 1])

with left:
    st.subheader("Live camera")
    camera_view = st.empty()
    prediction_view = st.empty()
    confidence_view = st.empty()
    camera_status = st.empty()

with right:
    st.subheader("Agent output")
    status_view = st.empty()
    sequence_view = st.empty()
    message_view = st.empty()
    question_view = st.empty()

st.subheader("Prediction history")
history_view = st.empty()

def render_agent_state():
    status_view.write(st.session_state.agent_status)

    if st.session_state.agent_sequence:
        sequence_view.write(
            " → ".join(st.session_state.agent_sequence)
        )
    else:
        sequence_view.caption("No confirmed signs yet.")

    if st.session_state.agent_message:
        message_view.success(st.session_state.agent_message)
    else:
        message_view.caption("No message drafted yet.")

    if st.session_state.agent_question:
        question_view.info(st.session_state.agent_question)
    else:
        question_view.empty()

    if st.session_state.history:
        history_view.dataframe(
            list(reversed(st.session_state.history[-20:])),
            use_container_width=True,
            hide_index=True,
        )
    else:
        history_view.caption("No predictions yet.")


render_agent_state()

if not st.session_state.camera_on:
    camera_status.info("Camera is stopped. Use Start camera in the sidebar.")
    st.stop()


camera = cv2.VideoCapture(int(camera_index))

if not camera.isOpened():
    st.session_state.camera_on = False
    st.error(
        "Could not open the webcam. Check camera permissions and "
        "make sure another application is not using it."
    )
    st.stop()

sequence = deque(maxlen=SEQUENCE_LENGTH)
missed_frames = 0
last_timestamp_ms = -1
last_draft_for_gesture = ""
last_confirmed_sign = None

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(LANDMARKER_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
)

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        camera_status.success("Camera is running. Show a sign to the camera.")

        while st.session_state.camera_on:
            ok, frame = camera.read()

            if not ok:
                camera_status.error("Unable to read a frame from the camera.")
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

                    if last_confirmed_sign is not None:
                        st.session_state.adapter.reset_gesture_lock()
                        last_confirmed_sign = None
                        last_draft_for_gesture = ""

            prediction = "Collecting frames"
            confidence = 0.0

            if len(sequence) == SEQUENCE_LENGTH and missed_frames == 0:
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

                prediction = (
                    best_label
                    if confidence >= CONFIDENCE_THRESHOLD
                    else "Uncertain"
                )

                if prediction not in ("Uncertain", "COLLECTING"):
                    event = make_prediction_event(prediction, confidence)

                    try:
                        agent_result = (
                            st.session_state.adapter.process_event(event)
                        )

                        seq_result = agent_result.get("sequence", {})

                        if seq_result.get("accepted"):
                            last_confirmed_sign = prediction

                            draft = agent_result.get("message") or {}
                            sentence = draft.get("sentence") or ""

                            if sentence and sentence != last_draft_for_gesture:
                                st.session_state.agent_message = sentence
                                last_draft_for_gesture = sentence

                            st.session_state.agent_status = (
                                f"Confirmed: {prediction}"
                            )

                            st.session_state.agent_sequence = list(
                                seq_result.get("gloss_buffer", [])
                            )

                            st.session_state.history.append(
                                {
                                    "time": time.strftime("%H:%M:%S"),
                                    "sign": prediction,
                                    "confidence": round(confidence, 3),
                                    "status": "Confirmed",
                                }
                            )

                        elif seq_result.get("reason"):
                            st.session_state.agent_status = (
                                f"{prediction}: {seq_result['reason']}"
                            )

                    except (ValueError, TypeError, KeyError) as exc:
                        st.session_state.agent_status = (
                            f"Agent event error: {exc}"
                        )

            st.session_state.last_prediction = prediction
            st.session_state.confidence = confidence

            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_view.image(
                display_frame,
                channels="RGB",
                use_container_width=True,
            )

            prediction_view.markdown(
                f"**Prediction:** {prediction}"
            )
            confidence_view.progress(
                max(0.0, min(1.0, confidence)),
                text=f"Confidence: {confidence:.1%}",
            )

            render_agent_state()

            time.sleep(0.03)

finally:
    camera.release()

    if st.session_state.camera_on:
        st.session_state.camera_on = False

    camera_status.info("Camera stopped.")
