
import json
from pathlib import Path

metadata_path = Path("data/processed/metadata.json")

with metadata_path.open(encoding="utf-8") as f:
    records = json.load(f)

total_detected = 0
total_frames = 0

print(f"{'Label':<10} {'Detected/Total':<16} {'Rate':>7}")
print("-" * 38)

for record in records:
    detected = record["detected_frames"]
    frames = record["total_frames"]
    rate = 100 * detected / max(1, frames)

    print(
        f"{record['label']:<10} "
        f"{detected:>4}/{frames:<11} "
        f"{rate:>5.1f}%"
    )

    total_detected += detected
    total_frames += frames

print("-" * 38)
overall = 100 * total_detected / max(1, total_frames)
print(f"Overall: {total_detected}/{total_frames} frames ({overall:.1f}%)")
