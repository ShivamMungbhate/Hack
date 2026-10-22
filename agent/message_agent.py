
import re


class MessageAgent:
    """Turns confirmed sign labels into a proposed message locally."""

    # Conservative templates: only map labels whose meanings are explicit.
    PHRASES = {
        ("HELLO",): "Hello.",
        ("HI",): "Hi.",
        ("THANK_YOU",): "Thank you.",
        ("THANKS",): "Thanks.",
        ("SORRY",): "Sorry.",
        ("YES",): "Yes.",
        ("NO",): "No.",
        ("HELP",): "I need help.",
        ("HELP", "PLEASE"): "Please help me.",
        ("I", "NEED", "HELP"): "I need help.",
        ("I", "AM", "FINE"): "I am fine.",
        ("HOW", "ARE", "YOU"): "How are you?",
        ("WHAT", "IS", "YOUR", "NAME"): "What is your name?",
        ("MY", "NAME", "IS"): None,  # Requires a name; ask for clarification.
        ("GOODBYE",): "Goodbye.",
        ("BYE",): "Bye.",
    }

    def draft(self, confirmed_signs, context=None):
        labels = [
            (item["sign"] if isinstance(item, dict) else str(item)).strip().upper()
            for item in confirmed_signs
        ]
        labels = [label for label in labels if label]

        base = {
            "gloss_buffer": labels,
            "sentence": None,
            "needs_confirmation": False,
            "clarification_question": None,
            "explanation": None,
        }

        if not labels:
            return {
                **base,
                "status": "needs_clarification",
                "clarification_question": "Please sign a message first.",
                "explanation": "No confirmed signs are available.",
            }

        key = tuple(labels)

        # Exact templates avoid inventing meaning from unknown labels.
        if key in self.PHRASES and self.PHRASES[key] is not None:
            return {
                **base,
                "status": "draft",
                "sentence": self.PHRASES[key],
                "needs_confirmation": True,
                "explanation": "Generated from a predefined local phrase template.",
            }

        # Support a few safe single-sign messages.
        if len(labels) == 1:
            label = labels[0]
            if re.fullmatch(r"[A-Z][A-Z0-9_ -]{0,39}", label):
                return {
                    **base,
                    "status": "needs_clarification",
                    "clarification_question": (
                        f"What do you mean by '{label.replace('_', ' ').lower()}'?"
                    ),
                    "explanation": (
                        "This sign has no configured phrase template."
                    ),
                }

        return {
            **base,
            "status": "needs_clarification",
            "clarification_question": (
                "Could you clarify the intended meaning of these signs?"
            ),
            "explanation": (
                "No matching local phrase template was found. "
                "No sentence was guessed."
            ),
        }
