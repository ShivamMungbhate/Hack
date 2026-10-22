from collections import deque
from .schemas import validate_prediction_event


class SequenceManager:
    """Deterministic sign stabilization and gloss-buffer management."""

    IGNORED = {"UNCERTAIN", "COLLECTING", "UNKNOWN", "NONE", ""}

    def __init__(self, confidence_threshold=0.30,
                 stable_observations=3, max_buffer=64):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Invalid confidence threshold")
        if stable_observations < 1 or max_buffer < 1:
            raise ValueError("Limits must be positive")

        self.threshold = confidence_threshold
        self.stable_observations = stable_observations
        self.max_buffer = max_buffer
        self.recent = deque(maxlen=stable_observations)
        self.buffer = []
        self.last_emitted = None

    @property
    def gloss_buffer(self):
        return [item["sign"] for item in self.buffer]

    def ingest(self, event):
        validate_prediction_event(event)

        sign = event["sign"].strip()
        confidence = event["confidence"]

        if sign.upper() in self.IGNORED or confidence < self.threshold:
            self.recent.clear()
            self.last_emitted = None
            return {
                "accepted": False,
                "reason": "uncertain_or_low_confidence",
                "gloss_buffer": self.gloss_buffer,
            }

        self.recent.append(sign)

        if len(self.recent) < self.stable_observations:
            return {
                "accepted": False,
                "reason": "stabilizing",
                "gloss_buffer": self.gloss_buffer,
            }

        if len({s.upper() for s in self.recent}) != 1:
            return {
                "accepted": False,
                "reason": "unstable",
                "gloss_buffer": self.gloss_buffer,
            }

        if sign.upper() == self.last_emitted:
            return {
                "accepted": False,
                "reason": "duplicate_suppressed",
                "gloss_buffer": self.gloss_buffer,
            }

        confirmed = {
            "sign": sign,
            "confidence": float(confidence),
            "timestamp_ms": event["timestamp_ms"],
        }

        self.buffer.append(confirmed)
        self.buffer = self.buffer[-self.max_buffer:]
        self.last_emitted = sign.upper()

        return {
            "accepted": True,
            "reason": "confirmed",
            "confirmed_sign": confirmed,
            "gloss_buffer": self.gloss_buffer,
        }

    def reset_gesture_lock(self):
        """Call when the frontend detects the end of a gesture."""
        self.recent.clear()
        self.last_emitted = None

    def remove_last(self):
        if not self.buffer:
            return None
        removed = self.buffer.pop()["sign"]
        self.reset_gesture_lock()
        return removed

    def clear(self):
        self.recent.clear()
        self.buffer.clear()
        self.last_emitted = None