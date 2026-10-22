'''
from .service import SignBridgeAgent


class AgentAdapter:
    """Connect prediction events to the agent without changing the API."""

    def __init__(self, agent=None):
        self.agent = agent or SignBridgeAgent()

    def process_event(self, event):
        """Consume one existing prediction event."""
        sequence = self.agent.ingest_prediction(event)

        message = None
        if sequence.get("accepted"):
            message = self.agent.draft_message()

        return {
            "sequence": sequence,
            "message": message,
        }

    def undo(self):
        return self.agent.undo()

    def clear(self):
        return self.agent.clear()

    def reset_gesture_lock(self):
        return self.agent.reset_gesture_lock()
'''

from .service import SignBridgeAgent


class AgentAdapter:
    """Connect prediction events to the SignBridge agent."""

    def __init__(self, agent=None):
        self.agent = agent or SignBridgeAgent()

    def process_event(self, event):
        """
        Process one prediction event.

        The sequence manager decides whether the prediction is stable
        enough to confirm. A message is generated only when a sign
        is accepted.
        """
        sequence = self.agent.ingest_prediction(event)

        message = None

        if sequence.get("accepted"):
            message = self.agent.draft_message()

        return {
            "sequence": sequence,
            "message": message,
        }

    def undo(self):
        """Remove the most recently confirmed sign."""
        return self.agent.undo()

    def clear(self):
        """Clear the current sign sequence."""
        return self.agent.clear()

    def reset_gesture_lock(self):
        """Allow the current or next gesture to be detected again."""
        return self.agent.reset_gesture_lock()
