from __future__ import annotations
import sys
from pathlib import Path
from . import VENDOR_COREAI_MODELS

PYTHON_SRC = VENDOR_COREAI_MODELS / "python" / "src"
SWIFT_PACKAGE = VENDOR_COREAI_MODELS


def ensure_python_path() -> Path:
	"""Insert ``coreai_models`` export package into ``sys.path``."""
	if not PYTHON_SRC.is_dir():
		raise FileNotFoundError(f"Missing vendored coreai-models python sources at {PYTHON_SRC}. "
		                        "Re-run vendor sync from README.")
	path = str(PYTHON_SRC)
	if path not in sys.path:
		sys.path.insert(0, path)
	return PYTHON_SRC


def swift_package_path() -> Path:
	if not (SWIFT_PACKAGE / "Package.swift").is_file():
		raise FileNotFoundError(f"Missing Package.swift in vendored coreai-models at {SWIFT_PACKAGE}")
	return SWIFT_PACKAGE
