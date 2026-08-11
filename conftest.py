"""Shared pytest configuration for local audio test dependencies."""
from __future__ import annotations

import os


os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
