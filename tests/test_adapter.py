
from agent.adapter import AgentAdapter


def make_event(sign="HELLO", confidence=0.95):
    return {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": 1000,
        "sign": sign,
        "confidence": confidence,
        "alternatives": [],
        "gloss_buffer": [],
        "sentence": None,
        "is_final": False,
    }


def test_adapter_accepts_existing_event_format():
    adapter = AgentAdapter()

    result = None
    for timestamp in (1000, 1100, 1200):
        event = make_event()
        event["timestamp_ms"] = timestamp
        result = adapter.process_event(event)

    assert result["sequence"]["accepted"] is True
    assert result["message"] is not None
    assert result["message"]["sentence"] == "Hello."


def test_adapter_does_not_draft_for_unconfirmed_sign():
    adapter = AgentAdapter()

    result = adapter.process_event(
        make_event(sign="Uncertain", confidence=0.1)
    )

    assert result["message"] is None
