# rebuild / gold / parity / speed. edit knobs, then: python trace.py --lib
import json, os, shutil, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YS = ROOT.parent
LIB = ROOT / "lib"
PY = "/opt/homebrew/Caskroom/miniforge/base/envs/yuna/bin/python"
ESPEAK_DATA_SRC = "/opt/homebrew/Cellar/espeak/1.48.04_1/share/espeak-data"
ESPEAK_LIB_SRC = "/opt/homebrew/opt/espeak/lib/libespeak.dylib"
PORTAUDIO_LIB_SRC = "/opt/homebrew/opt/portaudio/lib/libportaudio.2.dylib"
OPENJTALK_DICT_URL = "https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz"
OPENJTALK_SITE = "/opt/homebrew/Caskroom/miniforge/base/envs/yuna/lib/python3.12/site-packages/pyopenjtalk"
RAW = "/Users/yuki/Downloads/transcript-all.txt"
CLEANED = "/Users/yuki/Downloads/transcript-all.txt.cleaned"
LID = {"0": "en-us", "1": "ja", "2": "ru", "en-us": "en-us", "ja": "ja", "ru": "ru"}
DO = ["lib", "gold-live", "compare", "parity", "speed", "langs"]
LIVE = [("af", "other/af", "af", "af"), ("am", "test/am", "am", "am"), ("an", "europe/an", "an", "an"), ("as", "test/as", "as", "as"), ("az", "test/az", "az", "az"), ("bg", "europe/bg", "bg", "bg"), ("bn", "test/bn", "bn", "bn"), ("bs", "europe/bs", "hbs", "hr"), ("ca", "europe/ca", "ca", "ca"), ("cs", "europe/cs", "cs", "cs"), ("cy", "europe/cy", "cy", "cy"), ("da", "europe/da", "da", "da"), ("de", "de", "de", "de"), ("el", "europe/el", "el", "el"), ("en-us", "en-us", "en", "en-us"), ("eo", "other/eo", "eo", "eo"), ("es", "europe/es", "es", "es"), ("es-la", "es-la", "es", "es"), ("et", "europe/et", "et", "et"), ("eu", "test/eu", "eu", "eu"), ("fa", "asia/fa", "fa", "fa"), ("fa-pin", "asia/fa-pin", "fa", "fa"), ("fi", "europe/fi", "fi", "fi"), ("fr", "fr", "fr", "fr"), ("fr-be", "europe/fr-be", "fr", "fr"), ("ga", "europe/ga", "ga", "ga"), ("gd", "test/gd", "gd", "gd"), ("grc", "other/grc", "grc", "grc"), ("gu", "test/gu", "gu", "gu"), ("hi", "asia/hi", "hi", "hi"), ("hr", "europe/hr", "hbs", "hr"), ("hu", "europe/hu", "hu", "hu"), ("hy", "asia/hy", "hy", "hy"), ("hy-west", "asia/hy-west", "hy", "hy"), ("id", "asia/id", "id", "id"), ("is", "europe/is", "is", "is"), ("it", "europe/it", "it", "it"), ("jbo", "other/jbo", "jbo", "jbo"), ("ka", "asia/ka", "ka", "ka"), ("kl", "test/kl", "kl", "kl"), ("kn", "asia/kn", "kn", "kn"), ("ko", "test/ko", "ko", "ko"), ("ku", "asia/ku", "ku", "ku"), ("la", "other/la", "la", "la"), ("lfn", "other/lfn", "lfn", "lfn"), ("lt", "europe/lt", "lt", "lt"), ("lv", "europe/lv", "lv", "lv"), ("mk", "europe/mk", "mk", "mk"), ("ml", "asia/ml", "ml", "ml"), ("ms", "asia/ms", "ms", "ms"), ("nci", "test/nci", "nci", "nci"), ("ne", "asia/ne", "ne", "ne"), ("nl", "europe/nl", "nl", "nl"), ("no", "europe/no", "no", "no"), ("or", "test/or", "or", "or"), ("pa", "asia/pa", "pa", "pa"), ("pap", "test/pap", "pap", "pap"), ("pl", "europe/pl", "pl", "pl"), ("pt", "pt", "pt", "pt"), ("pt-pt", "europe/pt-pt", "pt", "pt"), ("ro", "europe/ro", "ro", "ro"), ("ru", "europe/ru", "ru", "ru"), ("si", "test/si", "si", "si"), ("sk", "europe/sk", "sk", "sk"), ("sl", "test/sl", "sl", "sl"), ("sq", "europe/sq", "sq", "sq"), ("sr", "europe/sr", "hbs", "hr"), ("sv", "europe/sv", "sv", "sv"), ("sw", "other/sw", "sw", "sw"), ("ta", "asia/ta", "ta", "ta"), ("te", "test/te", "te", "te"), ("tr", "asia/tr", "tr", "tr"), ("ur", "test/ur", "ur", "ur"), ("vi", "asia/vi", "vi", "vi"), ("vi-hue", "asia/vi-hue", "vi", "vi"), ("vi-sgn", "asia/vi-sgn", "vi", "vi"), ("zh", "asia/zh", "zh", "zh"), ("zh-yue", "asia/zh-yue", "zhy", "zh-yue"), ]  # lid, cellar voice relpath, dict stem, owner lid (shared dicts symlink to owner)
CASES = [("EN01", "en-us", 'Wait… she said — "don\'t" (please) — «now», “then”: ready?!'), ("EN02", "en-us", "¡Hello! ¿Ready? {whisper} (aside) [note]"), ("EN03", "en-us", "Hold on... then . . . go… stay."), ("EN04", "en-us", "See pages 3–5, then 3-7, then 3 - 9, then 3—11."), ("EN05", "en-us", "Pay $1, $2, $1.5K, 99€, £10, ¥500, 7₽, and 25$."), ("EN06", "en-us", "About 1,234,567 plus 2.5e-3 and 10K versus 3.14."), ("EN07", "en-us", "Dr. J. A. Smith met Mrs. Jones at St. Mary Blvd. at 3 p.m.; he ain't ready, e.g. now."), ("EN08", "en-us", "Read chapter XII, not IV, nor DC, nor I, after AD 12."), ("EN09", "en-us", "See https://example.com/a and www.test.org *now*."), ("EN10", "en-us", "The window-pane cracked; twenty-one left."), ("EN11", "en-us", "Stop; wait: now, then."), ("EN12", "en-us", "What???? No!!!! Ready!? Done?!"), ("EN13", "en-us", "“Yes,” she said, \"no,\" and ‟maybe.\""), ("EN14", "en-us", "Is it pretty because the women went to the city?"), ("EN15", "en-us", "Love—a temporary confluence (really): \"yes\"—no…"), ("RU01", "ru", "«Ну что?!» — сказала она (тихо)… “ладно”, 11₽ и 3-7, не 2.5e3."), ("RU02", "ru", "Это (конечно) так?! Нет!!"), ("RU03", "ru", "Дай 1₽, 2₽, 5₽, 11₽, 21₽, 22₽ и 1.5₽."), ("RU04", "ru", "Мерцание звёзд—это сердцебиение Вселенной."), ("RU05", "ru", "Тонкий серп луны похож на чей-то загадочный прищуренный древний глаз. Он молчаливо наблюдает за нашими смешными суетливыми ночными земными делами. Лунный свет обнажает скрытые инстинкты: которые আমরা тщательно прячем днем."), ("RU06", "ru", "Москва сказала OK, hello, мир."), ("JA01", "ja", "本当に、「待って…」と（静かに）言ったんです——やっぱり100％?! ３-５円。"), ("JA02", "ja", "本当ですか？ そうである。"), ("JA03", "ja", "「待って！」と『彼女』は言った。"), ("JA04", "ja", "やっぱり日本へ行ったんです。100％です！"), ("JA05", "ja", "散歩と天気とこんにちはと本当とあんみつ。"), ("JA06", "ja", "それは $5 と ¥100 です。"), ("JA07", "ja", "OK、分かった。"), ("XX-DE", "de", "Guten Tag, wie geht es Ihnen?"), ("XX-FR", "fr", "Bonjour, comment allez-vous?"), ("XX-ES", "es", "¡Hola! ¿Cómo estás?"), ("XX-ZH", "zh", "你好，世界。"), ("XX-BN", "bn", "আমরা যাচ্ছি।"), ]
BANNED_VOICES = ("voices/en", "voices/default", "voices/other/en-n", "voices/other/en-rp", "voices/other/en-sc", "voices/other/en-wi", "voices/other/en-wm")


def _copy(src, dst):
	src, dst = Path(src), Path(dst)
	if src.is_symlink(): src = src.resolve()
	if not src.exists(): raise RuntimeError(f"missing {src}")
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.is_symlink() or dst.is_file(): dst.unlink()
	elif dst.is_dir(): shutil.rmtree(dst)
	if src.is_dir(): shutil.copytree(src, dst, symlinks=False)
	else: shutil.copy2(src, dst)
	print(f"  {dst.relative_to(ROOT)}  {dst.stat().st_size}")


def _refuse_links(root):
	left = [p for p in Path(root).rglob("*") if p.is_symlink()]
	if left: raise RuntimeError("REFUSE symlink " + " ".join(str(p.relative_to(ROOT)) for p in left[:8]))


def _pin_dylib(core):
	espeak, pa = core / "libespeak.dylib", core / "libportaudio.2.dylib"
	if not espeak.is_file() or espeak.is_symlink(): raise RuntimeError("libespeak.dylib missing")
	os.chmod(espeak, 0o644)
	if pa.is_file(): os.chmod(pa, 0o644)
	subprocess.check_call(["install_name_tool", "-id", "@loader_path/libespeak.dylib", str(espeak)])
	if pa.is_file():
		subprocess.check_call(["install_name_tool", "-id", "@loader_path/libportaudio.2.dylib", str(pa)])
		subprocess.check_call(["install_name_tool", "-change", "/opt/homebrew/opt/portaudio/lib/libportaudio.2.dylib", "@loader_path/libportaudio.2.dylib", str(espeak)])
	subprocess.check_call(["codesign", "--force", "--sign", "-", str(espeak)])
	if pa.is_file(): subprocess.check_call(["codesign", "--force", "--sign", "-", str(pa)])


def flatten_lib():
	print("flatten lib/ (copy every symlink)")
	if not LIB.is_dir(): raise RuntimeError("lib/ missing")
	links = [p for p in LIB.rglob("*") if p.is_symlink()]
	for p in sorted(links, key=lambda x: (x.is_dir(), len(x.parts))):
		tgt = p.resolve()
		if not tgt.exists(): raise RuntimeError(f"broken {p} -> {tgt}")
		_copy(tgt, p)
	pa = Path(PORTAUDIO_LIB_SRC)
	if pa.is_file() and not (LIB / "libportaudio.2.dylib").is_file(): _copy(pa.resolve(), LIB / "libportaudio.2.dylib")
	_pin_dylib(LIB)
	_refuse_links(LIB)
	print(f"  ok  {len(links)} copied  0 symlinks")


def unite_lib():
	print("unite lib/  (dylib + espeak-data/voices/<lid> + ja/)")
	old = LIB / "_espeak"
	live = LIB / "espeak-data"
	if (old / "espeak-data").is_dir() and not live.exists(): (old / "espeak-data").rename(live)
	for name in ("libespeak.dylib", "libportaudio.2.dylib", "speak_lib.h"):
		src = old / name
		if src.is_file(): _copy(src, LIB / name)
	voices = live / "voices"
	for lid, vrel, dname, owner in LIVE:
		src = voices / vrel
		dst = voices / lid
		if src.is_file() and src.resolve() != dst.resolve(): _copy(src, dst)
		elif not dst.is_file() and (LIB / lid / "voice").is_file(): _copy(LIB / lid / "voice", dst)
		if lid == owner and not (live / f"{dname}_dict").is_file() and (LIB / lid / "dict").is_file(): _copy(LIB / lid / "dict", live / f"{dname}_dict")
	for nest in ("asia", "europe", "other", "test"):
		p = voices / nest
		if p.is_dir(): shutil.rmtree(p)
	for p in voices.rglob("*"):
		if p.is_symlink() or (p.is_file() and p.name.endswith(" 2")): p.unlink()
	for lid, vrel, dname, owner in LIVE:
		p = LIB / lid
		if p.is_dir(): shutil.rmtree(p)
	if old.exists(): shutil.rmtree(old)
	for junk in ("phontab", "phondata", "phonindex", "intonations"):
		p = LIB / junk
		if p.is_file(): p.unlink()
	_pin_dylib(LIB)
	if not (live / "phontab").is_file(): raise RuntimeError("lib/espeak-data/phontab missing")
	if not (voices / "en-us").is_file() or not (voices / "ru").is_file(): raise RuntimeError("flat voices missing")
	if (LIB / "ja" / "open_jtalk_dic_utf_8-1.11").is_symlink(): raise RuntimeError("ja dict still a symlink")
	_refuse_links(LIB)
	print(f"  voices {len(list(voices.iterdir()))}  dicts {len(list(live.glob('*_dict')))}  0 _espeak  0 lang packs")


def build_lib():
	src = Path(ESPEAK_DATA_SRC)
	if not src.is_dir(): raise RuntimeError(f"missing {src}")
	live = LIB / "espeak-data"
	print("build lib/ + lib/espeak-data/voices/<lid>  (copies only)")
	hdr_keep = None
	for hp in (LIB / "speak_lib.h", LIB / "_espeak" / "speak_lib.h"):
		if hp.is_file() and not hp.is_symlink():
			hdr_keep = hp.read_bytes()
			break
	if live.exists(): shutil.rmtree(live)
	old = LIB / "_espeak"
	if old.exists(): shutil.rmtree(old)
	dylib = Path(ESPEAK_LIB_SRC)
	if not dylib.is_file(): raise RuntimeError("libespeak.dylib missing")
	_copy(dylib.resolve(), LIB / "libespeak.dylib")
	pa = Path(PORTAUDIO_LIB_SRC)
	if pa.is_file(): _copy(pa.resolve(), LIB / "libportaudio.2.dylib")
	hdr = ROOT / "vendor" / "espeak-src" / "src" / "speak_lib.h"
	if hdr.is_file(): _copy(hdr, LIB / "speak_lib.h")
	elif hdr_keep is not None: (LIB / "speak_lib.h").write_bytes(hdr_keep)
	_pin_dylib(LIB)
	for f in ("phontab", "phondata", "phonindex", "intonations"):
		_copy(src / f, live / f)
	for lid, vrel, dname, owner in LIVE:
		_copy(src / "voices" / vrel, live / "voices" / lid)
		if lid == owner: _copy(src / f"{dname}_dict", live / f"{dname}_dict")
	for bad in BANNED_VOICES:
		p = live / bad
		if p.exists(): raise RuntimeError(f"REFUSE live {bad}")
	if (live / "voices" / "!v").exists() or (live / "voices" / "mb").exists(): raise RuntimeError("REFUSE !v/mbrola")
	site = Path(OPENJTALK_SITE) / "open_jtalk_dic_utf_8-1.11"
	jdst = LIB / "ja" / "open_jtalk_dic_utf_8-1.11"
	jdst.parent.mkdir(parents=True, exist_ok=True)
	if jdst.is_symlink(): jdst.unlink()
	if not (jdst / "sys.dic").is_file():
		if jdst.exists(): shutil.rmtree(jdst)
		if (site / "sys.dic").is_file(): _copy(site, jdst)
		else:
			tgz = LIB / "ja" / "open_jtalk_dic_utf_8-1.11.tar.gz"
			if not tgz.is_file(): urllib.request.urlretrieve(OPENJTALK_DICT_URL, tgz)
			with tarfile.open(tgz) as t:
				t.extractall(LIB / "ja")
	_refuse_links(LIB)
	print(f"  langs {len(LIVE)}  live dicts {len(list(live.glob('*_dict')))}  0 symlinks")


def _load_text_py():
	import importlib.util
	tp = YS / "text.py"
	spec = importlib.util.spec_from_file_location("yuna_text_ref", tp)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def gold_live():
	print("capture gold from live text.py")
	ref = _load_text_py()
	out = []
	for cid, lid, src in CASES:
		row = {"id": cid, "lid": lid, "src": src, "raw": ref.text_to_phonemes_raw(src, lid), "clean": ref.text_cleaners(src, lid)}
		out.append(row)
		print(f"  {cid} {lid} raw={row['raw'][:60]!r}")
	p = ROOT / "data" / "gold-cases.json"
	p.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
	print(f"  wrote {p}  {len(out)}")


def compare():
	print("compare phonemize vs gold-cases + live text.py")
	sys.path.insert(0, str(ROOT))
	from phonemize import text_to_phonemes_raw, text_cleaners
	gold = json.loads((ROOT / "data" / "gold-cases.json").read_text(encoding="utf-8"))
	ref = None
	try:
		ref = _load_text_py()
		if not hasattr(ref, "backend"): ref = None
	except Exception as e:
		print(f"  live text.py skipped ({type(e).__name__}: {e})")
		ref = None
	bad = []
	for row in gold:
		got_r, got_c = text_to_phonemes_raw(row["src"], row["lid"]), text_cleaners(row["src"], row["lid"])
		if row["lid"] in ("en-us", "ru", "ja"):
			if got_r != row["raw"] or got_c != row["clean"]:
				bad.append(row["id"])
				print(f"  GOLD MISS {row['id']}\n    src {row['src']!r}\n    got_r {got_r!r}\n    gold_r {row['raw']!r}\n    got_c {got_c!r}\n    gold_c {row['clean']!r}")
		elif row["lid"] in ("de", "fr", "es", "zh", "bn"):
			if not got_r or got_r == row["src"] and row["lid"] != "zh":
				print(f"  LANG {row['id']} {row['lid']} {got_r!r}")
		if ref is not None and row["lid"] in ("en-us", "ru", "ja"):
			lv_r, lv_c = ref.text_to_phonemes_raw(row["src"], row["lid"]), ref.text_cleaners(row["src"], row["lid"])
			if got_r != lv_r or got_c != lv_c:
				bad.append(row["id"] + "-live")
				print(f"  LIVE MISS {row['id']}\n    got {got_r!r}\n    live {lv_r!r}")
	print(f"  GOLD {len(gold) - len([x for x in bad if not x.endswith('-live')])}/{len(gold)}  misses={bad}")
	if any(not x.endswith("-live") for x in bad if x.split("-")[0] in ("EN", "RU", "JA") or x[:2] in ("EN", "RU", "JA")): raise SystemExit("compare failed")
	core = [x for x in bad if x.startswith(("EN", "RU", "JA"))]
	if core: raise SystemExit(f"compare failed {core}")
	print("COMPARE OK")


def lang_of(row):
	if len(row) >= 4 and row[2] in LID: return LID[row[2]]
	path = row[0].lower()
	if "-ja" in path or "/ja" in path or "wavs-ja" in path: return "ja"
	if "-ru" in path or "/ru" in path or "wavs-ru" in path: return "ru"
	return "en-us"


def parity(limit=0, live=False):
	print("parity vs transcript-all.txt.cleaned")
	sys.path.insert(0, str(ROOT))
	from phonemize import load_filepaths_and_text, text_to_phonemes_raw, _SYMBOL_TO_ID
	ref = None
	if live:
		ref = _load_text_py()
		assert ref._SYMBOL_TO_ID == _SYMBOL_TO_ID, "177-id table drifted"
	raw_rows = load_filepaths_and_text(RAW)
	clean_rows = load_filepaths_and_text(CLEANED)
	n = min(len(raw_rows), len(clean_rows))
	if limit: n = min(n, limit)
	bad_clean, bad_live = [], []
	by = {"en-us": 0, "ru": 0, "ja": 0}
	t0 = time.perf_counter()
	for i in range(n):
		row, gold = raw_rows[i], clean_rows[i][-1]
		lang = lang_of(row)
		got = text_to_phonemes_raw(row[-1], lang)
		by[lang] += 1
		if got != gold:
			bad_clean.append(i)
			if len(bad_clean) <= 8: print(f"  CLEAN MISS [{i}] {lang}\n    src  {row[-1]!r}\n    got  {got!r}\n    gold {gold!r}")
		if ref is not None and hasattr(ref, "text_to_phonemes_raw"):
			if getattr(ref, "backend", None) is not None:
				lv = ref.text_to_phonemes_raw(row[-1], lang)
				if got != lv:
					bad_live.append(i)
					if len(bad_live) <= 8: print(f"  LIVE MISS [{i}] {lang}\n    got  {got!r}\n    live {lv!r}")
		if (i + 1) % 2000 == 0 or i + 1 == n: print(f"  {i + 1}/{n}  clean_miss={len(bad_clean)}  {(time.perf_counter() - t0):.1f}s")
	print("  per-lang", by)
	print(f"  CLEAN {n - len(bad_clean)}/{n}" + (f"  LIVE {n - len(bad_live)}/{n}" if bad_live or (ref and getattr(ref, 'backend', None)) else "") + f"  {(time.perf_counter() - t0):.2f}s")
	if bad_clean: raise SystemExit(f"parity failed clean={len(bad_clean)}")
	if bad_live: print(f"  LIVE {len(bad_live)} ɪ/ᵻ flips vs isolated text.py (filelist stream is the gold)")
	print("PARITY OK")


def speed():
	print("speed bench")
	sys.path.insert(0, str(ROOT))
	from phonemize import load_filepaths_and_text, text_to_phonemes_raw, text_cleaners
	from phonemize import _g2p
	raw_rows = load_filepaths_and_text(RAW)
	samples = {"en-us": [], "ru": [], "ja": []}
	for row in raw_rows:
		lang = lang_of(row)
		if len(samples[lang]) < 400: samples[lang].append(row[-1])
		if all(len(v) == 400 for v in samples.values()): break
	for lang, xs in samples.items():  # warmup
		text_to_phonemes_raw(xs[0], lang)
	for lang, xs in samples.items():
		t0 = time.perf_counter()
		for s in xs:
			text_to_phonemes_raw(s, lang)
		dt = time.perf_counter() - t0
		print(f"  raw  {lang:5}  {len(xs)} utt  {dt:.3f}s  {len(xs)/dt:.1f} utt/s  {1000*dt/len(xs):.2f} ms/utt")
	en = samples["en-us"][:200]
	t0 = time.perf_counter()
	for s in en:
		text_cleaners(s, "en-us")
	dt = time.perf_counter() - t0
	print(f"  clean en-us {len(en)}  {dt:.3f}s  {len(en)/dt:.1f} utt/s")
	try:  # live phonemizer if still present
		ref = _load_text_py()
		if getattr(ref, "backend", None) is None: raise RuntimeError("already swapped")
		ref.text_to_phonemes_raw(en[0], "en-us")
		t0 = time.perf_counter()
		for s in en:
			ref.text_to_phonemes_raw(s, "en-us")
		dt = time.perf_counter() - t0
		print(f"  live raw en-us {len(en)}  {dt:.3f}s  {len(en)/dt:.1f} utt/s")
	except Exception as e:
		print(f"  live bench skipped ({e})")


def langs_smoke():
	print("all-lang SetVoiceByName")
	sys.path.insert(0, str(ROOT))
	from phonemize import text_to_phonemes_raw, _VOICES
	probe = {"en-us": "Hello, world!", "ru": "Привет, мир!", "de": "Guten Tag.", "fr": "Bonjour.", "es": "Hola.", "it": "Ciao.", "zh": "你好。", "bn": "আমরা।", "ja": "こんにちは。", "pl": "Cześć.", "pt": "Olá.", "ko": "안녕하세요."}
	for lid, s in probe.items():
		got = text_to_phonemes_raw(s, lid)
		print(f"  {lid:6}  {got!r}")
		if lid != "ja" and (not got or got == s) and lid not in ("zh", ): print(f"  WARN {lid} looks identity")
	n = 0
	for lid in sorted(_VOICES):
		got = text_to_phonemes_raw("1", lid)
		if got is None: raise RuntimeError(lid)
		n += 1
	print(f"  voices ok {n}/{len(_VOICES)}")


def _ibridge(a, b):
	if a == b: return True
	if len(a) != len(b): return False
	return all(x == y or {x, y} == {"ɪ", "ᵻ"} for x, y in zip(a, b))


def official():
	print("official phonemizer / pyopenjtalk vs yuna")  # live official engines — not text.py (that is yuna now)
	os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = ESPEAK_LIB_SRC
	from phonemizer.backend import EspeakBackend
	EspeakBackend.set_library(ESPEAK_LIB_SRC)
	kw = dict(preserve_punctuation=True, with_stress=True, language_switch="remove-flags", words_mismatch="warn")
	off_en, off_ru = EspeakBackend("en-us", **kw), EspeakBackend("ru", **kw)

	def off_espeak(text, lid):
		be = off_en if lid == "en-us" else off_ru
		got = be.phonemize([text.strip()], strip=False)
		ph = (got[0] if got else "").strip()
		return ph if ph or lid != "ru" else text

	import pyopenjtalk
	sys.path.insert(0, str(ROOT))
	from phonemize import text_to_phonemes_raw, japanese_to_ipa3, extract_fullcontext as yuna_labels

	gold = json.loads((ROOT / "data" / "gold-cases.json").read_text(encoding="utf-8"))
	print("\n== gold-cases (official live vs yuna vs stored gold) ==")
	for row in gold:
		cid, lid, src = row["id"], row["lid"], row["src"]
		yun = text_to_phonemes_raw(src, lid)
		if lid in ("en-us", "ru"):
			off = off_espeak(src, lid)
			ok_oy, ok_og, ok_yg = off == yun, off == row["raw"], yun == row["raw"]
			tag = "OK" if ok_oy and ok_yg else ("ɪ/ᵻ" if _ibridge(off, yun) and _ibridge(yun, row["raw"]) else "MISS")
			if tag != "OK": print(f"  {tag} {cid} off={off!r}\n         yun={yun!r}\n         gld={row['raw']!r}")
			else: print(f"  OK   {cid}")
		elif lid == "ja":
			import phonemize as P
			real = P.extract_fullcontext
			P.extract_fullcontext = pyopenjtalk.extract_fullcontext
			off = P.japanese_to_ipa3(src.strip()).strip()
			P.extract_fullcontext = real
			g2p = pyopenjtalk.g2p(src)
			ok = off == yun == row["raw"]
			print(f"  {'OK' if ok else 'MISS'} {cid}  g2p={g2p!r}\n         yun={yun!r}" + ("" if ok else f"\n         off={off!r}\n         gld={row['raw']!r}"))
		else:
			yun = text_to_phonemes_raw(src, lid)
			print(f"  SMOKE {cid} {lid} yun={yun[:50]!r}")

	print("\n== CLI espeak --ipa (NOT the yuna API) ==")
	import subprocess
	for s in ("Hello, world!", "Wait… she said — don't."):
		cli = subprocess.check_output(["/opt/homebrew/opt/espeak/bin/espeak", "-q", "--ipa", "-b", "1", "-v", "en-us", "--path=/opt/homebrew/opt/espeak/share", s], text=True).strip()
		yun = text_to_phonemes_raw(s, "en-us")
		print(f"  src {s!r}\n    cli {cli!r}\n    yun {yun!r}\n    match {cli == yun}")

	print("\n== filelist official vs yuna vs gold ==")
	from phonemize import load_filepaths_and_text
	raw_rows, clean_rows = load_filepaths_and_text(RAW), load_filepaths_and_text(CLEANED)
	n = min(len(raw_rows), len(clean_rows))
	score = {k: {"n": 0, "oy": 0, "yg": 0, "og": 0, "ib": 0} for k in ("en-us", "ru", "ja")}
	shown = 0
	t0 = time.perf_counter()
	for i in range(n):
		row, goldv = raw_rows[i], clean_rows[i][-1]
		lid = lang_of(row)
		src = row[-1]
		yun = text_to_phonemes_raw(src, lid)
		if lid == "ja":
			import phonemize as P
			real = P.extract_fullcontext
			P.extract_fullcontext = pyopenjtalk.extract_fullcontext
			off = P.japanese_to_ipa3(src.strip()).strip()
			P.extract_fullcontext = real
		else:
			off = off_espeak(src, lid)
		sc = score[lid]
		sc["n"] += 1
		if off == yun: sc["oy"] += 1
		elif _ibridge(off, yun): sc["ib"] += 1
		if yun == goldv: sc["yg"] += 1
		if off == goldv: sc["og"] += 1
		if off != yun and not _ibridge(off, yun) and shown < 8:
			print(f"  MISS [{i}] {lid}\n    src {src!r}\n    off {off!r}\n    yun {yun!r}\n    gld {goldv!r}")
			shown += 1
		if (i + 1) % 10000 == 0 or i + 1 == n: print(f"  {i + 1}/{n}  {(time.perf_counter() - t0):.1f}s")
	for lid, sc in score.items():
		print(f"  {lid:5} n={sc['n']}  yuna==gold {sc['yg']}/{sc['n']}  official==gold {sc['og']}/{sc['n']}  official==yuna {sc['oy']}/{sc['n']}  ɪ/ᵻ-only {sc['ib']}")

	print("\n== ja labels official extract_fullcontext vs yuna ==")
	lab_n = lab_ok = 0
	for i in range(n):
		if lang_of(raw_rows[i]) != "ja": continue
		src = raw_rows[i][-1]
		if not src.strip(): continue  # only JP spans yuna sends to the frontend — compare on full strings that are mostly JP
		a, b = pyopenjtalk.extract_fullcontext(src), yuna_labels(src)
		lab_n += 1
		if a == b: lab_ok += 1
		if lab_n >= 400: break
	print(f"  labels equal {lab_ok}/{lab_n} (first {lab_n} ja rows that both accepted)")

	print("\n== speed (400/lid, warmup 1) ==")
	from phonemize import espeak_ipa as yuna_espeak
	samples = {"en-us": [], "ru": [], "ja": []}
	for row in raw_rows:
		lid = lang_of(row)
		if len(samples[lid]) < 400: samples[lid].append(row[-1])
		if all(len(v) == 400 for v in samples.values()): break
	yuna_espeak(samples["en-us"][0], "en-us")
	off_espeak(samples["en-us"][0], "en-us")
	yuna_labels(samples["ja"][0])
	pyopenjtalk.extract_fullcontext(samples["ja"][0])
	for lid in ("en-us", "ru"):
		xs = samples[lid]
		t0 = time.perf_counter()
		for s in xs:
			off_espeak(s, lid)
		dt0 = time.perf_counter() - t0
		t0 = time.perf_counter()
		for s in xs:
			yuna_espeak(s, lid)
		dt1 = time.perf_counter() - t0
		print(f"  {lid:5} official {len(xs)/dt0:.0f} utt/s   yuna {len(xs)/dt1:.0f} utt/s   {dt0/dt1:.2f}×")
	xs = samples["ja"]
	t0 = time.perf_counter()
	for s in xs:
		pyopenjtalk.extract_fullcontext(s)
	dt0 = time.perf_counter() - t0
	t0 = time.perf_counter()
	for s in xs:
		yuna_labels(s)
	dt1 = time.perf_counter() - t0
	print(f"  ja    official extract_fullcontext {len(xs)/dt0:.0f} utt/s   yuna {len(xs)/dt1:.0f} utt/s")
	t0 = time.perf_counter()
	for s in xs:
		text_to_phonemes_raw(s, "ja")
	dt2 = time.perf_counter() - t0
	print(f"  ja    yuna full ipa3 {len(xs)/dt2:.0f} utt/s  (g2p is a different string, not timed as a match)")


def dump_unicode():
	print("dump unidecode tables")
	from unidecode import unidecode as _u
	import unidecode as U
	table = {}
	for cp in range(0x110000):
		ch = chr(cp)
		repl = _u(ch)
		if repl != ch: table[f"{cp:04X}"] = repl
	meta = {"_meta": {"unidecode": getattr(U, "__version__", "?"), "count": len(table)}}
	out = ROOT / "data" / "unicode.json"
	out.write_text(json.dumps({**meta, **table}, ensure_ascii=False, indent=0, sort_keys=True) + "\n", encoding="utf-8")
	print(f"  unicode.json  {out.stat().st_size}  {len(table)}")


if __name__ == "__main__":
	args = [a[2:] if a.startswith("--") else a for a in (sys.argv[1:] or DO)]
	if "lib" in args: build_lib()
	if "flatten" in args: flatten_lib()
	if "unite" in args: unite_lib()
	if "unicode" in args: dump_unicode()
	if "gold-live" in args: gold_live()
	if "compare" in args: compare()
	if "langs" in args: langs_smoke()
	if "official" in args: official()
	if "speed" in args: speed()
	if any(a == "parity" or a.startswith("parity=") or a == "parity-live" for a in args):
		lim = 0
		for a in args:
			if a.startswith("parity="): lim = int(a.split("=", 1)[1])
		parity(lim, live=("parity-live" in args))
