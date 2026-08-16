#!/usr/bin/env python3
"""
verify_parity.py
-----------------
Proves that localization_service.run_localization() produces numerically
IDENTICAL results to localization/inference.py::localize() (the function
the CLI, localize.py, calls). Run this after any change to
localization_service.py.

Usage:
    python webapp/backend/verify_parity.py
"""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "..", "localization"))

from inference import localize  # noqa: E402
from localization_service import run_localization  # noqa: E402

OUTPUTS_DIR = os.path.join(ROOT, "..", "..", "outputs")


def run_service(reference_path, search_path, topk=5):
    final = None
    for ev in run_localization(reference_path, search_path, topk=topk):
        if ev["stage"] == "result":
            final = ev["data"]["result"]
    return final


def main():
    cases = ["dram_018", "dram_020", "finfet_005", "finfet_009"]
    all_ok = True
    for case in cases:
        ann_path = os.path.join(OUTPUTS_DIR, "annotations", f"{case}.json")
        ann = json.load(open(ann_path))
        ref_path = os.path.join(OUTPUTS_DIR, ann["reference_path"])
        search_path = os.path.join(OUTPUTS_DIR, ann["search_path"])

        cli_result = localize(ref_path, search_path)
        svc_result = run_service(ref_path, search_path)

        keys = ["x", "y", "confidence", "ambiguous", "candidates_considered"]
        ok = all(cli_result[k] == svc_result[k] for k in keys)
        status = "OK" if ok else "MISMATCH"
        if not ok:
            all_ok = False
        print(f"[{status}] {case}")
        print(f"    CLI:     {{ {', '.join(f'{k}={cli_result[k]}' for k in keys)} }}")
        print(f"    Service: {{ {', '.join(f'{k}={svc_result[k]}' for k in keys)} }}")

    print()
    if all_ok:
        print("PARITY OK: web service matches CLI exactly on all test cases.")
        sys.exit(0)
    else:
        print("PARITY FAILURE: web service diverges from CLI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
