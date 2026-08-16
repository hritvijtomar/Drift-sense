#!/usr/bin/env python3
"""
Top-level entry point matching the recommended submission structure
(PS Section 5). Thin wrapper around localization/inference.py -- this is
the script that must "process a pair or evaluator-provided batch without
manual source-code changes" (Section 4.C).

Usage:
    python localize.py --reference REF.png --search SEARCH.png
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "localization"))
from inference import main

if __name__ == "__main__":
    main()
