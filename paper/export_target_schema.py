#!/usr/bin/env python3
"""Export the exact ordered 192-score inventory from pinned release metadata."""

import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
snap = root / "evidence" / "snapshots"
train = json.loads((snap / "training_normalization.json").read_text())
display = json.loads((snap / "display_normalization.json").read_text())
names = train["score_names"]
assert len(names) == len(display["targets"]) == 192


def family(index):
    for stop, name in (
        (40, "EmoNet emotion"), (97, "VoiceNet dimension"),
        (100, "Genuineness / blend / legacy quality"),
        (119, "Empathic Insight extra"), (123, "AudioBox"),
        (130, "DNSMOS"), (131, "Burst count"),
        (192, "VoiceCLAP attribute"),
    ):
        if index < stop:
            return name
    raise AssertionError(index)


with (root / "target_schema.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("index", "family", "target_name", "training_mean",
                     "training_std", "display_median", "display_std",
                     "display_reference", "display_reference_n"))
    for i, name in enumerate(names):
        item = display["targets"][i]
        assert item["index"] == i and item["name"] == name
        writer.writerow((i, family(i), name, train["score_mean"][i],
                         train["score_std"][i], item["median"], item["std"],
                         item["reference"], item["n"]))

print("Wrote 192 ordered target rows to target_schema.csv")
