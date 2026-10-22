
from .message_agent import MessageAgent
from .sequence_manager import SequenceManager
from .schemas import validate_prediction_event


class SignBridgeAgent:
    """Coordinates prediction validation, sign stabilization, and drafting."""

    def __init__(self, sequence_manager=None, message_agent=None):
        self.sequence_manager = sequence_manager or SequenceManager()
        self.message_agent = message_agent or MessageAgent()

    def ingest_prediction(self, event):
        validate_prediction_event(event)
        return self.sequence_manager.ingest(event)

    def draft_message(self, context=None):
        signs = self.sequence_manager.gloss_buffer
        return self.message_agent.draft(signs, context=context)

    def undo(self):
        return self.sequence_manager.remove_last()

    def clear(self):
        return self.sequence_manager.clear()

    def reset_gesture_lock(self):
        return self.sequence_manager.reset_gesture_lock()
