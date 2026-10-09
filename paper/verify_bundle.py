#!/usr/bin/env python3
"""Check frozen evidence, cited keys, and the checkpoint identity of paper tables.

Run from any directory with Python 3; no third-party packages or source audio needed.
"""

from __future__ import annotations

import hashlib
import json
import re
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
PAPER = (ROOT / "main.tex").read_text()
SNAP = ROOT / "evidence" / "snapshots"


def load(name: str) -> dict:
    return json.loads((SNAP / name).read_text())


def has(value: float, places: int, suffix: str = "") -> None:
    rendered = f"{value:.{places}f}{suffix}"
    assert rendered in PAPER, f"Paper is missing evidence value {rendered}"


for line in (ROOT / "evidence" / "SHA256SUMS").read_text().splitlines():
    digest, relative = line.split(maxsplit=1)
    candidate = WORKSPACE / relative
    if not candidate.exists():
        candidate = ROOT / Path(relative).relative_to("humaneness_ears_paper")
    content = candidate.read_bytes()
    assert hashlib.sha256(content).hexdigest() == digest, relative

base = load("base_public_metrics.json")
medium = load("small_public_metrics.json")
for source in (base, medium):
    has(source["emonet"]["all_mapped_40"]["pearson"], 3)
for dataset in ("crema", "ravdess"):
    pair = "/".join(f"{source[dataset]['accuracy'] * 100:.1f}" for source in (base, medium))
    assert f"{pair}\\%" in PAPER, f"Paper is missing fixed-crosswalk pair {pair}"

for prefix in ("base", "small"):
    for dataset, table_digits in (("crema", 2), ("ravdess", 2)):
        adapter = load(f"{prefix}_{dataset}_matched_adapter.json")
        assert "192 raw predictions" in adapter["protocol"]
        assert "actor-disjoint" in adapter["protocol"]
        has(adapter["metrics"]["accuracy"] * 100, table_digits, "\\%")
    tuned = load(f"{prefix}_gemini_test_metrics.json")
    assert tuned["clips"] == 3544
    has(tuned["frame_f1"], 3)
    has(tuned["burst_localization"]["0.5"]["f1"], 3)
    before = load(f"{prefix}_legacy_p3_test_metrics.json")
    after = load(f"{prefix}_gemini_p3_test_metrics.json")
    assert before["clips"] == after["clips"] == 2000
    assert before["frame_f1"] > after["frame_f1"]
    assert before["burst_localization"]["0.5"]["f1"] > after["burst_localization"]["0.5"]["f1"]

bib = (ROOT / "references.bib").read_text()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
used = set()
for command in re.findall(r"\\cite(?:p|t)?\{([^}]+)\}", PAPER):
    used.update(part.strip() for part in command.split(","))
assert used <= keys, f"Undefined BibTeX keys: {used - keys}"
assert "pre-Gemini" in PAPER and "post-Gemini" in PAPER
assert "an earlier, separate pre-Gemini" in PAPER.lower() or "An earlier, separate pre-Gemini" in PAPER
assert "two epochs" in PAPER

normalization = load("training_normalization.json")
mapping = load("gemini_burst_mapping.json")
with (ROOT / "target_schema.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == len(normalization["score_names"]) == 192
assert [row["target_name"] for row in rows] == normalization["score_names"]
assert len(mapping["checkpoint_classes"]) == 53

print(f"Verified {len(used)} cited references, evidence checksums, and final-model result tables.")
