"""
ASGI shim: run from repo root with:
  uvicorn main:app --reload

Loads the FastAPI app from StressDetection/main.py (inner package folder).
"""
import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_inner_dir = _root / "StressDetection"
if str(_inner_dir) not in sys.path:
    sys.path.insert(0, str(_inner_dir))

_inner = _inner_dir / "main.py"
_spec = importlib.util.spec_from_file_location("stressdetection_fastapi_inner_main", _inner)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
app = _mod.app
