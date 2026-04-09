from __future__ import annotations

import os

# Stabilize CI/local unittest runtime when native libs are present.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
