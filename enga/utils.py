"""Shared utilities: deterministic hashing, seeding, JSON I/O, logging."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np


def stable_hash(text: str) -> int:
    """Salt-free deterministic hash (Python's builtin hash() is salted per process)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python/numpy RNGs and return a numpy Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Logger:
    """Minimal prefix logger (tqdm-friendly: writes to stderr)."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(f"[enga] {msg}", flush=True)
