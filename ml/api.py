
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
