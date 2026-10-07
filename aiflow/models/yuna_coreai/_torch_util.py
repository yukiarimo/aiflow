from __future__ import annotations


def ensure_torch():
	"""Import torch, with a clear message when macOS blocks iCloud paths."""
	try:
		import torch

		return torch
	except PermissionError as exc:
		raise SystemExit("\nPyTorch failed to import: macOS blocked file access (PermissionError).\n"
		                 "This often happens when aiflow or model weights live under iCloud Drive "
		                 "(~/Library/Mobile Documents/...).\n\n"
		                 "Fix (pick one):\n"
		                 "  1. System Settings → Privacy & Security → Full Disk Access → "
		                 "enable Terminal (or iTerm)\n"
		                 "  2. Clone aiflow outside iCloud and `pip install -e` from that copy\n"
		                 "  3. Copy model weights to ~/Downloads or ~/Documents and pass that "
		                 "--model-path\n") from exc
