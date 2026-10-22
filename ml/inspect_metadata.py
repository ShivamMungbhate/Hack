import json
from collections import defaultdict
from pathlib import Path

records = json.loads(
    Path("data/processed/metadata.json").read_text(encoding="utf-8")
)

stats = defaultdict(lambda: [0, 0, 0])

for item in records:
    label = item["label"]
    detected = item["detected_frames"]
    total = item["total_frames"]
    stats[label][0] += detected
    stats[label][1] += total
    stats[label][2] += 1

print("Detection rate by sign:")
for label, (detected, total, count) in sorted(stats.items()):
    rate = detected / total if total else 0
    print(f"{label:<8} {detected:>4}/{total:<4} "
          f"({rate:.1%}) across {count} clips")

print("\nClips with the lowest detection rates:")
for item in sorted(
    records,
    key=lambda x: x["detected_frames"] / max(x["total_frames"], 1)
)[:10]:
    rate = item["detected_frames"] / max(item["total_frames"], 1)
    print(f"{item['label']:<8} {rate:.1%}  {item['video']}")