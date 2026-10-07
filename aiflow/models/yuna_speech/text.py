if __package__:  # thin re-export of yuna_phonemizer. callers stay on aiflow.models.yuna_speech.text
	from .yuna_phonemizer import *  # noqa: F401, F403
	from .yuna_phonemizer import _SYMBOL_TO_ID, _symbol_to_id, _id_to_symbol, SPACE_ID, symbols
else:
	import importlib.util, sys
	from pathlib import Path
	_yp = Path(__file__).resolve().parent / "yuna_phonemizer"
	_name = "_yuna_phonemizer_impl"
	if _name not in sys.modules:
		_spec = importlib.util.spec_from_file_location(_name, _yp / "__init__.py", submodule_search_locations=[str(_yp)])
		_mod = importlib.util.module_from_spec(_spec)
		sys.modules[_name] = _mod
		_spec.loader.exec_module(_mod)
	from _yuna_phonemizer_impl import *  # noqa: F401, F403
	from _yuna_phonemizer_impl import _SYMBOL_TO_ID, _symbol_to_id, _id_to_symbol, SPACE_ID, symbols
