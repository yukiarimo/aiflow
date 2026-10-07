from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VENDOR_COREAI_MODELS = PACKAGE_ROOT / "vendor" / "coreai-models"
__all__ = ["PACKAGE_ROOT", "VENDOR_COREAI_MODELS"]
