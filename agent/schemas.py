def validate_prediction_event(event):
    required = {
        "schema_version", "type", "timestamp_ms", "sign",
        "confidence", "alternatives", "gloss_buffer",
        "sentence", "is_final",
    }
    if not isinstance(event, dict) or not required.issubset(event):
        raise ValueError("Invalid prediction event fields")

    if event["schema_version"] != "1.0":
        raise ValueError("Unsupported schema version")
    if event["type"] != "sign_prediction":
        raise ValueError("Unsupported event type")
    if type(event["timestamp_ms"]) is not int or event["timestamp_ms"] < 0:
        raise ValueError("Invalid timestamp")
    if not isinstance(event["sign"], str) or not event["sign"].strip():
        raise ValueError("Invalid sign")
    if type(event["confidence"]) not in (int, float):
        raise ValueError("Invalid confidence")
    if not 0 <= event["confidence"] <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    if not isinstance(event["alternatives"], list):
        raise ValueError("alternatives must be a list")
    if not isinstance(event["gloss_buffer"], list):
        raise ValueError("gloss_buffer must be a list")
    if event["sentence"] is not None and not isinstance(event["sentence"], str):
        raise ValueError("sentence must be a string or null")
    if type(event["is_final"]) is not bool:
        raise ValueError("is_final must be a boolean")