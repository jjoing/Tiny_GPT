#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run train.py once for each of the four custom novel datasets.

Convenience wrapper for the assignment deliverable: produces one checkpoint,
loss-curve PNG, loss-history JSON, and generated text sample per book in
results/, all named after the book so they're easy to drop into the README.
Any extra arguments (e.g. --device cuda) are forwarded to every run.
"""

import subprocess
import sys

DATASETS = [
    "alice_in_wonderland",
    "frankenstein",
    "pride_and_prejudice",
    "sherlock_holmes",
]

COMMON_ARGS = [
    "--block-size", "128",
    "--batch-size", "64",
    "--emb-dim", "128",
    "--num-heads", "4",
    "--num-layers", "4",
    "--epochs", "20",
    "--max-steps", "200",
    "--lr", "3e-4",
    "--train-fraction", "0.9",
    "--sample-every", "5",
    "--start-text", "The",
]


def main() -> None:
    extra_args = sys.argv[1:]
    for name in DATASETS:
        cmd = [
            sys.executable, "train.py",
            "--dataset", name,
            "--run-name", name,
            *COMMON_ARGS,
            *extra_args,
        ]
        print("=" * 80)
        print("running:", " ".join(cmd))
        print("=" * 80)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
