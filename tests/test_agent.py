
import pytest

from agent.sequence_manager import SequenceManager
from agent.service import SignBridgeAgent
from agent.message_agent import MessageAgent


def event(sign="HELLO", confidence=0.95, timestamp=1000):
    return {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": timestamp,
        "sign": sign,
        "confidence": confidence,
        "alternatives": [],
        "gloss_buffer": [],
        "sentence": None,
        "is_final": False,
    }


def test_stabilizes_sign():
    manager = SequenceManager(stable_observations=3)

    assert not manager.ingest(event(timestamp=1))["accepted"]
    assert not manager.ingest(event(timestamp=2))["accepted"]
    assert manager.ingest(event(timestamp=3))["accepted"]

    assert manager.gloss_buffer == ["HELLO"]


def test_suppresses_duplicate_predictions():
    manager = SequenceManager(stable_observations=2)

    manager.ingest(event(timestamp=1))
    manager.ingest(event(timestamp=2))

    result = manager.ingest(event(timestamp=3))

    assert result["reason"] == "duplicate_suppressed"
    assert manager.gloss_buffer == ["HELLO"]


def test_uncertain_prediction_resets_duplicate_lock():
    manager = SequenceManager(stable_observations=1)

    assert manager.ingest(event())["accepted"]
    manager.ingest(event("Uncertain", 0.1, 1001))

    assert manager.ingest(event(timestamp=1002))["accepted"]
    assert manager.gloss_buffer == ["HELLO", "HELLO"]


def test_low_confidence_is_ignored():
    manager = SequenceManager(stable_observations=1)

    result = manager.ingest(event("HELP", 0.1))

    assert not result["accepted"]
    assert manager.gloss_buffer == []


def test_invalid_event_is_rejected():
    manager = SequenceManager()

    with pytest.raises(ValueError):
        manager.ingest({"sign": "HELLO"})


def test_clear_and_undo():
    manager = SequenceManager(stable_observations=1)
    manager.ingest(event())

    assert manager.remove_last() == "HELLO"
    assert manager.remove_last() is None

    manager.ingest(event("YES", timestamp=2000))
    manager.clear()

    assert manager.gloss_buffer == []


def test_message_agent_uses_local_templates():
    agent = MessageAgent()

    result = agent.draft(["HELLO"])

    assert result["status"] == "draft"
    assert result["sentence"] == "Hello."
    assert result["needs_confirmation"] is True
    assert result["gloss_buffer"] == ["HELLO"]


def test_empty_buffer_needs_clarification():
    agent = MessageAgent()

    result = agent.draft([])

    assert result["status"] == "needs_clarification"
    assert result["sentence"] is None
    assert result["needs_confirmation"] is False
    assert result["gloss_buffer"] == []
    assert result["clarification_question"]


def test_unknown_sign_needs_clarification():
    agent = MessageAgent()

    result = agent.draft(["UNKNOWN_SIGN"])

    assert result["status"] == "needs_clarification"
    assert result["sentence"] is None
    assert result["needs_confirmation"] is False
    assert result["gloss_buffer"] == ["UNKNOWN_SIGN"]


def test_facade_connects_sequence_to_local_message_agent():
    service = SignBridgeAgent(
        sequence_manager=SequenceManager(stable_observations=1),
        message_agent=MessageAgent(),
    )

    service.ingest_prediction(event("HELLO"))
    result = service.draft_message()

    assert result["status"] == "draft"
    assert result["sentence"] == "Hello."
    assert result["gloss_buffer"] == ["HELLO"]
    assert result["needs_confirmation"] is True


def test_facade_clear_removes_buffer():
    service = SignBridgeAgent(
        sequence_manager=SequenceManager(stable_observations=1),
        message_agent=MessageAgent(),
    )

    service.ingest_prediction(event("HELLO"))
    service.clear()

    result = service.draft_message()

    assert result["status"] == "needs_clarification"
    assert result["gloss_buffer"] == []
