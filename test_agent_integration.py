
from agent.service import SignBridgeAgent


def make_event(sign, timestamp):
    return {
        "schema_version": "1.0",
        "type": "sign_prediction",
        "timestamp_ms": timestamp,
        "sign": sign,
        "confidence": 0.99,
        "alternatives": [],
        "gloss_buffer": [],
        "sentence": None,
        "is_final": False,
    }


agent = SignBridgeAgent()

# Repeat HELLO to satisfy the sequence stabilizer.
for timestamp in range(1, 10):
    result = agent.ingest_prediction(make_event("HELLO", timestamp))
    print("Ingest:", result)

print("\nDraft:")
print(agent.draft_message())
