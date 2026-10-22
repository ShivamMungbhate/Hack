'''
import json
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "signbridge_svm.joblib")
labels = json.loads((ROOT / "models" / "sign_labels.json").read_text(encoding="utf-8"))
sessions = {}

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            session = body.get("session", "default")
            frame = np.asarray(body["features"], dtype=np.float32)

            if frame.shape != (126,):
                raise ValueError("Expected 126 features per frame")

            sequence = sessions.setdefault(session, deque(maxlen=30))
            sequence.append(frame)

            result = {"label": "COLLECTING", "confidence": 0.0}
            if len(sequence) == 30:
                x = np.stack(sequence)
                features = np.concatenate(
                    [x.mean(axis=0), x.std(axis=0), x[-1] - x[0]]
                ).reshape(1, -1)
                probs = model.predict_proba(features)[0]
                i = int(np.argmax(probs))
                confidence = float(probs[i])
                result = {
                    "label": str(model.classes_[i]) if confidence >= 0.30 else "Uncertain",
                    "confidence": confidence,
                }

            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, *_):
        pass

print("Python sign model API running at http://127.0.0.1:8000")
HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
'''


import json
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(ROOT / "models" / "signbridge_svm.joblib")
labels = json.loads(
    (ROOT / "models" / "sign_labels.json").read_text(encoding="utf-8")
)

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126
CONFIDENCE_THRESHOLD = 0.30
sessions = {}


def make_event(
    sign,
    confidence=0.0,
    alternatives=None,
    gloss_buffer=None,
    sentence=None,
    is_final=False,
):
    """Create an event matching prediction-event.schema.json."""
    return {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": int(time.time() * 1000),
        "sign": str(sign),
        "confidence": float(confidence),
        "alternatives": alternatives or [],
        "gloss_buffer": gloss_buffer or [],
        "sentence": sentence,
        "is_final": bool(is_final),
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, data):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/predict", "/reset"):
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Invalid request body size")

            body = json.loads(self.rfile.read(length))
            session = body.get("session", "default")

            if not isinstance(session, str) or not session:
                raise ValueError("session must be a non-empty string")
            if len(session) > 128:
                raise ValueError("session is too long")

            if self.path == "/reset":
                sessions.pop(session, None)
                self.send_json(200, {"status": "reset", "session": session})
                return

            frame = np.asarray(body.get("features"), dtype=np.float32)
            if frame.shape != (FEATURE_COUNT,):
                raise ValueError("Expected 126 features per frame")
            if not np.isfinite(frame).all():
                raise ValueError("Features must contain only finite numbers")

            sequence = sessions.setdefault(
                session, deque(maxlen=SEQUENCE_LENGTH)
            )
            sequence.append(frame)

            if len(sequence) < SEQUENCE_LENGTH:
                event = make_event("COLLECTING")
                self.send_json(200, event)
                return

            x = np.stack(sequence)
            features = np.concatenate(
                [x.mean(axis=0), x.std(axis=0), x[-1] - x[0]]
            ).reshape(1, -1)

            probabilities = model.predict_proba(features)[0]
            class_names = [str(c) for c in model.classes_]
            order = np.argsort(probabilities)[::-1]

            best_index = int(order[0])
            confidence = float(probabilities[best_index])
            predicted_sign = class_names[best_index]

            if confidence < CONFIDENCE_THRESHOLD:
                sign = "Uncertain"
                alternative_indices = order[:3]
            else:
                sign = predicted_sign
                alternative_indices = [
                    i for i in order if int(i) != best_index
                ][:3]

            alternatives = [
                {
                    "sign": class_names[int(i)],
                    "confidence": float(probabilities[int(i)]),
                }
                for i in alternative_indices
            ]

            event = make_event(
                sign=sign,
                confidence=confidence,
                alternatives=alternatives,
            )
            self.send_json(200, event)

        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            self.send_json(500, {"error": "Internal prediction error"})

    def log_message(self, *_):
        pass


print("SignBridge API running at http://127.0.0.1:8000")
HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
