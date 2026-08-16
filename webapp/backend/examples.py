#!/usr/bin/env python3
"""
examples.py
-----------
Registry of curated demo cases for the Drift-Sense hero flow. Every
entry is backed by a real annotation file under outputs/annotations/
and real image files under outputs/reference/ and outputs/search/ --
nothing here is synthesized.

Each entry's ground-truth center/bbox comes straight from the
annotation JSON, so the frontend's "ground truth vs predicted" overlay
and the failure-classification panel are always showing real
evaluation data, never fabricated numbers.
"""
import os
import json

OUTPUTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs"
)
ANNOTATIONS_DIR = os.path.join(OUTPUTS_DIR, "annotations")

# (annotation_id, short display label, one-line curator's note, expected category)
# "expected category" is a curator's note based on evaluation/results.json
# and evaluation/diagnose_dram.py -- the UI verifies it against the LIVE
# backend result on every run rather than trusting this label blindly.
# Selected by inspecting evaluation/results.json (see webapp/README.md
# "How demo examples were chosen" for the full rationale):
CURATED = [
    ("finfet_005", "Obvious Success",
     "Clean FinFET match: correct, unambiguous, no near-tie candidates.",
     "success"),
    ("finfet_009", "FinFET Success (Flagged Ambiguous)",
     "Correct result, but the backend still raises its ambiguity flag -- "
     "shows the flag isn't the same thing as 'wrong'.",
     "success"),
    ("dram_002", "DRAM / Periodic Pattern",
     "Periodic DRAM array resolved correctly despite several "
     "near-identical NCC peaks at the repeat pitch.",
     "success"),
    ("dram_018", "Candidate-Generation Miss",
     "Ground truth never entered the top-5 NCC candidate set -- "
     "the failure happened upstream of ranking.",
     "candidate_generation_miss"),
    ("dram_020", "Ranking / Selection Failure",
     "Ground truth WAS a top-5 candidate (rank 4) but lost to another "
     "candidate by a 0.215 score margin -- not a near tie.",
     "ranking_selection_failure"),
]


def _load_annotation(ann_id):
    path = os.path.join(ANNOTATIONS_DIR, f"{ann_id}.json")
    with open(path) as f:
        return json.load(f)


def list_examples():
    out = []
    for ann_id, label, note, expected_category in CURATED:
        try:
            ann = _load_annotation(ann_id)
        except FileNotFoundError:
            continue
        ref_file = os.path.join(OUTPUTS_DIR, ann["reference_path"])
        search_file = os.path.join(OUTPUTS_DIR, ann["search_path"])
        if not (os.path.exists(ref_file) and os.path.exists(search_file)):
            continue
        out.append({
            "id": ann_id,
            "label": label,
            "note": note,
            "expected_category": expected_category,
            "style": ann["style"],
            "pair_id": ann["pair_id"],
            "hard_case": bool(ann.get("hard_case", False)),
            "heavier_noise": bool(ann.get("heavier_noise", False)),
            "reference_url": f"/api/media/reference/{os.path.basename(ref_file)}",
            "search_url": f"/api/media/search/{os.path.basename(search_file)}",
            "ground_truth": {
                "x": ann["center_x"],
                "y": ann["center_y"],
                "bbox": ann["bbox"],
            },
        })
    return out


def get_example(ann_id):
    for ann_id_, label, note, expected_category in CURATED:
        if ann_id_ == ann_id:
            ann = _load_annotation(ann_id)
            ref_file = os.path.join(OUTPUTS_DIR, ann["reference_path"])
            search_file = os.path.join(OUTPUTS_DIR, ann["search_path"])
            return {
                "id": ann_id,
                "label": label,
                "reference_path": ref_file,
                "search_path": search_file,
                "ground_truth": (ann["center_x"], ann["center_y"]),
            }
    return None
