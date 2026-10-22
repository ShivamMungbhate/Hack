
import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8") as f:
        return json.load(f)


def test_label_ids_are_unique_and_idle_is_zero():
    config = load_json("configs/labels.json")
    labels = config["labels"]

    ids = [item["id"] for item in labels]
    names = [item["name"] for item in labels]

    assert len(ids) == len(set(ids)), "Duplicate label IDs"
    assert len(names) == len(set(names)), "Duplicate label names"
    assert config["idle_label"] == "IDLE"
    assert labels[0] == {"id": 0, "name": "IDLE"}
    assert sorted(ids) == list(range(len(labels)))


def test_expected_vocabulary_is_present():
    config = load_json("configs/labels.json")
    actual = {item["name"] for item in config["labels"]}

    expected = {
        "IDLE", "HELLO", "THANK-YOU", "PLEASE", "YES", "NO",
        "HELP", "NAME", "ME", "YOU", "WATER", "FOOD", "GOOD",
        "SORRY", "STOP", "WHERE", "WHAT", "HOW", "WANT", "NEED",
        "FINISHED"
    }

    assert actual == expected


def test_landmark_feature_count():
    hand_features = 2 * 21 * 3
    pose_features = 33 * 4

    assert hand_features == 126
    assert pose_features == 132
    assert hand_features + pose_features == 258


def test_sample_prediction_event_matches_schema():
    schema = load_json("contracts/prediction-event.schema.json")

    event = {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": 1792400000000,
        "sign": "HELLO",
        "confidence": 0.94,
        "alternatives": [
            {"sign": "HELLO", "confidence": 0.94},
            {"sign": "THANK-YOU", "confidence": 0.03},
            {"sign": "PLEASE", "confidence": 0.01}
        ],
        "gloss_buffer": ["HELLO"],
        "sentence": "Hello!",
        "is_final": True
    }

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(event)


def test_invalid_confidence_is_rejected():
    schema = load_json("contracts/prediction-event.schema.json")

    event = {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": 1000,
        "sign": "HELLO",
        "confidence": 1.5,
        "alternatives": [],
        "gloss_buffer": [],
        "sentence": None,
        "is_final": False
    }

    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(event)), (
        "Confidence above 1.0 should fail validation"
    )
