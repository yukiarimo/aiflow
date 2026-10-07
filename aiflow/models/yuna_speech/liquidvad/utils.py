import json
import os
import sys
from glob import glob
import numpy as np
import torch

SAMPLE_RATE = 16000
HOP_LENGTH = 160
N_MELS = 128
RMVPE_DIM = 3
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(ROOT, 'data')
DEFAULT_RUNS = os.path.join(ROOT, 'runs')
DEFAULT_OUT = os.path.join(ROOT, 'out')
DEFAULT_CKPT = os.path.join(DEFAULT_RUNS, 'liquid_vad.pt')
NOISE_WAV = os.path.join(DEFAULT_DATA, 'noise.wav')
DEFAULT_RMVPE_DIR = '/Users/yuki/Library/Mobile Documents/com~apple~CloudDocs/Personal/Github/yuna-ai/lib/models/rmvpe'
F0_MIN = 50.0
F0_MAX = 800.0


def get_device(name='mps'):
	if name == 'cpu':
		return torch.device('cpu')
	if name == 'cuda' and torch.cuda.is_available():
		return torch.device('cuda')
	if torch.backends.mps.is_available():
		return torch.device('mps')
	return torch.device('cpu')


def resolve_rmvpe(path=None):
	path = path or DEFAULT_RMVPE_DIR
	if path.endswith('.mlpackage') or os.path.isfile(path):
		return path
	ml = os.path.join(path, 'model.mlpackage')
	pt = os.path.join(path, 'model.pt')
	if sys.platform == 'darwin' and os.path.exists(ml):
		return ml
	if os.path.isfile(pt):
		return pt
	if os.path.exists(ml):
		return ml
	raise FileNotFoundError('RMVPE not found at %s' % path)


def load_rmvpe(path=None, device=None):
	from aiflow.models.yuna_speech.rmvpe.models import load_coreml_model, load_model
	device = device or get_device()
	path = resolve_rmvpe(path)
	if path.endswith('.mlpackage'):
		return {'model': load_coreml_model(path), 'kind': 'coreml', 'device': device, 'path': path}
	model, kind = load_model(path, device)
	return {'model': model, 'kind': kind, 'device': device, 'path': path}


def load_audio(path, sr=SAMPLE_RATE):
	from aiflow.models.yuna_speech.rmvpe.models import load_audio_16k
	audio = load_audio_16k(path, sr)
	return np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)


def save_audio(path, audio, sr=SAMPLE_RATE):
	import soundfile as sf
	os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
	sf.write(path, np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0), sr, subtype='PCM_16')


def to_16k(audio, sr, target=SAMPLE_RATE):
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	if audio.size == 0 or int(sr) == int(target):
		return audio
	from aiflow.models.yuna_audio.audio_np import resample_audio
	return np.asarray(resample_audio(audio, sr, target), dtype=np.float32)


def audio_peak(audio):
	audio = np.asarray(audio, dtype=np.float32)
	if audio.size == 0:
		return 0.0
	return float(np.max(np.abs(audio)))


def is_silent(audio, thresh=1e-4):
	return audio_peak(audio) < thresh


def wav_is_silent(path, thresh=1e-4):
	if not os.path.isfile(path):
		return True
	import soundfile as sf
	x, _ = sf.read(path, always_2d=False, dtype='float32')
	return is_silent(x, thresh)


def n_frames_for(n_samples, hop=HOP_LENGTH):
	return 1 + int(n_samples) // hop


def sec_to_frame(t, hop=HOP_LENGTH, sr=SAMPLE_RATE):
	return int(round(float(t) * sr / hop))


def frame_to_sec(i, hop=HOP_LENGTH, sr=SAMPLE_RATE):
	return float(i) * hop / sr


def _log_f0(f0):
	out = np.zeros_like(f0, dtype=np.float32)
	ok = f0 > 0
	if np.any(ok):
		lo, hi = np.log(F0_MIN), np.log(F0_MAX)
		out[ok] = (np.log(np.clip(f0[ok], F0_MIN, F0_MAX)) - lo) / (hi - lo)
	return out


def extract_mel(audio, hop=HOP_LENGTH):
	from aiflow.models.yuna_speech.rmvpe.models import audio_to_mel_rvc
	audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
	if audio.size < HOP_LENGTH * 4:
		audio = np.pad(audio, (0, HOP_LENGTH * 4 - audio.size))
	mel = audio_to_mel_rvc(audio).numpy().astype(np.float32)
	mel = (mel - mel.mean()) / (mel.std() + 1e-5)
	return {'mel': mel, 'n_frames': int(mel.shape[-1])}


def extract_features(audio, rmvpe, hop=HOP_LENGTH):
	from aiflow.models.yuna_speech.rmvpe.models import audio_to_mel_rvc, infer_rvc, infer_rvc_coreml
	from aiflow.models.yuna_speech.rmvpe.utils import salience_to_hz
	audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
	if audio.size < HOP_LENGTH * 4:
		audio = np.pad(audio, (0, HOP_LENGTH * 4 - audio.size))
	mel = audio_to_mel_rvc(audio).numpy().astype(np.float32)
	if rmvpe['kind'] == 'coreml':
		salience = infer_rvc_coreml(rmvpe['model'], audio).numpy()
	else:
		wav = torch.from_numpy(audio)
		salience = infer_rvc(rmvpe['model'], wav, rmvpe['device']).detach().cpu().numpy()
	t = min(mel.shape[-1], salience.shape[0])
	mel = mel[:, :t]
	salience = salience[:t]
	f0 = salience_to_hz(salience, thred=0.2).astype(np.float32)
	voiced = (f0 > 0).astype(np.float32)
	peak = salience.max(axis=-1).astype(np.float32)
	mel = (mel - mel.mean()) / (mel.std() + 1e-5)
	rmvpe_feat = np.stack([_log_f0(f0), voiced, np.clip(peak, 0.0, 1.0)], axis=0)
	return {'mel': mel, 'rmvpe': rmvpe_feat, 'f0': f0, 'n_frames': t}


def event_bump(n, idx, sigma=3.0):
	t = np.arange(n, dtype=np.float32)
	return np.exp(-0.5 * ((t - float(idx)) / sigma)**2)


def make_labels(utterances, n_frames, hop=HOP_LENGTH, sr=SAMPLE_RATE, sigma=3.0):
	speech = np.zeros(n_frames, dtype=np.float32)
	start = np.zeros(n_frames, dtype=np.float32)
	stop = np.zeros(n_frames, dtype=np.float32)
	for u in utterances:
		a = int(np.clip(sec_to_frame(u['start'], hop, sr), 0, n_frames - 1))
		b = int(np.clip(sec_to_frame(u['stop'], hop, sr), 0, n_frames - 1))
		if b < a:
			a, b = b, a
		speech[a:b + 1] = 1.0
		start = np.maximum(start, event_bump(n_frames, a, sigma))
		stop = np.maximum(stop, event_bump(n_frames, b, sigma))
	return {'speech': speech, 'start': start, 'stop': stop}


def find_onset(audio, sr, t0, t1, hop=HOP_LENGTH):
	x0, x1 = int(max(0, t0) * sr), int(max(t0, t1) * sr)
	x = np.asarray(audio[x0:x1], dtype=np.float32)
	if x.size < hop * 4:
		return float(t0)
	win = 512
	n = 1 + max(0, (len(x) - win) // hop)
	rms = np.empty(n, dtype=np.float32)
	for i in range(n):
		w = x[i * hop:i * hop + win]
		rms[i] = np.sqrt(np.mean(w * w) + 1e-12)
	head = max(1, min(n, int(0.4 * sr / hop)))
	noise = float(np.median(rms[:head]))
	thr = max(0.01, noise * 3.0, float(np.percentile(rms, 20)) * 2.2)
	need = max(3, int(0.04 * sr / hop))
	run = 0
	for i, v in enumerate(rms):
		if v >= thr:
			run += 1
			if run >= need:
				i0 = max(0, i - need - 2)
				return t0 + i0 * hop / sr
		else:
			run = 0
	if rms.max() > thr * 0.45:
		return t0 + int(np.argmax(rms)) * hop / sr
	return float(t0)


def _peaks(x, thr, min_dist):
	idx = np.where(x >= thr)[0]
	keep = []
	for i in idx:
		lo, hi = max(0, i - min_dist), min(len(x), i + min_dist + 1)
		if x[i] < x[lo:hi].max():
			continue
		if keep and i - keep[-1] < min_dist:
			if x[i] > x[keep[-1]]:
				keep[-1] = i
			continue
		keep.append(int(i))
	return keep


def _mask_segments(prob, thr, hop_s, min_dur, exit_th=None):
	enter = float(thr)
	leave = float(thr if exit_th is None else exit_th)
	on, start, segs = False, 0, []
	for i, v in enumerate(prob):
		if not on and v >= enter:
			on, start = True, i
		elif on and v < leave:
			if (i - start) * hop_s >= min_dur:
				segs.append((start * hop_s, i * hop_s))
			on = False
	if on and (len(prob) - start) * hop_s >= min_dur:
		segs.append((start * hop_s, len(prob) * hop_s))
	return segs


def _snap(t, peaks, window):
	if not peaks:
		return t
	cands = [p for p in peaks if abs(p - t) <= window]
	if not cands:
		return t
	return float(min(cands, key=lambda p: abs(p - t)))


def decode_segments(speech_p, start_p, stop_p, hop=HOP_LENGTH, sr=SAMPLE_RATE, speech_th=0.45, start_th=0.30, stop_th=0.35, min_dur=0.20, max_dur=20.0):
	"""Speech islands first (the dips you see), then snap edges to start/stop peaks."""
	speech_p = np.asarray(speech_p, dtype=np.float32)
	start_p = np.asarray(start_p, dtype=np.float32)
	stop_p = np.asarray(stop_p, dtype=np.float32)
	hop_s = hop / sr
	min_dist = max(2, int(0.35 / hop_s))
	islands = _mask_segments(speech_p, speech_th, hop_s, min_dur, exit_th=max(0.14, speech_th - 0.30))
	starts = [i * hop_s for i in _peaks(start_p, start_th, max(2, int(0.25 / hop_s)))]
	stops = [i * hop_s for i in _peaks(stop_p, stop_th, min_dist)]
	stops_soft = [i * hop_s for i in _peaks(stop_p, min(0.18, stop_th), min_dist)]
	segs = []
	for a, b in islands:
		ib = max(1, int(round(b / hop_s)))
		ia = max(0, int(round(a / hop_s)))
		head = start_p[ia:min(len(start_p), ia + max(1, int(0.25 / hop_s)))]
		if a <= 0.22 and (b - a) < 0.90:
			continue
		if a <= 0.20 and (b - a) < 1.15 and float(head.max() if head.size else 0.0) < 0.20:
			continue
		a = _snap(a, starts, 0.40)
		b2 = _snap(b, stops, 0.45)
		b = b2 if b2 != b else _snap(b, stops_soft, 0.35)
		later = [s for s in stops if b < s <= b + 1.20]
		nxt = [s for s in starts if s > b]
		if later and (not nxt or later[-1] < nxt[0] - 0.15):
			i0, i1 = int(round(b / hop_s)), int(round(later[-1] / hop_s))
			if float(speech_p[max(0, i0):min(len(speech_p), i1 + 1)].max()) >= 0.50:
				b = later[-1]
		if a <= 0.22 and (b - a) < 0.90:
			continue
		dur = b - a
		if min_dur <= dur <= max_dur:
			if segs and a <= segs[-1][1] + 0.22:
				segs[-1] = (segs[-1][0], max(segs[-1][1], b))
			else:
				segs.append((float(a), float(b)))
	if not segs and stops:
		cursor = 0
		for b in _peaks(stop_p, stop_th, min_dist):
			active = np.where(speech_p[cursor:b] >= speech_th)[0]
			if active.size == 0:
				continue
			a = cursor + int(active[0])
			if min_dur <= (b - a) * hop_s <= max_dur:
				segs.append((a * hop_s, b * hop_s))
				cursor = b + 1
	return segs


def takes_dir(data_dir):
	return os.path.join(data_dir, 'takes')


def shown_dir(data_dir):
	return os.path.join(data_dir, 'shown')


def resolve_ckpt(path=None):
	path = path or DEFAULT_CKPT
	if os.path.isfile(path):
		return path
	legacy = os.path.join(DEFAULT_RUNS, 'yuna_vad.pt')
	if os.path.abspath(path) == os.path.abspath(DEFAULT_CKPT) and os.path.isfile(legacy):
		return legacy
	return path


def cache_dir(data_dir):
	return os.path.join(data_dir, 'cache')


def list_take_jsons(data_dir):
	return sorted(glob(os.path.join(takes_dir(data_dir), 'take_*.json')))


def read_json(path):
	with open(path) as f:
		return json.load(f)


def write_json(path, obj):
	os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
	with open(path, 'w') as f:
		json.dump(obj, f, indent=2)


def cache_path_for(json_path):
	data_dir = os.path.dirname(os.path.dirname(os.path.abspath(json_path)))
	stem = os.path.splitext(os.path.basename(json_path))[0]
	return os.path.join(cache_dir(data_dir), stem + '.npz')


def cache_take(meta_or_path, rmvpe, hop=HOP_LENGTH, sr=SAMPLE_RATE):
	meta = read_json(meta_or_path) if isinstance(meta_or_path, str) else meta_or_path
	wav = meta.get('wav') or meta['path']
	if isinstance(meta_or_path, str):
		base = os.path.dirname(os.path.abspath(meta_or_path))
		if not os.path.isfile(wav):
			wav = os.path.join(base, os.path.basename(meta['path']))
		out = cache_path_for(meta_or_path)
	else:
		out = meta['cache']
	audio = load_audio(wav, sr)
	feats = extract_features(audio, rmvpe, hop)
	labels = make_labels(meta.get('utterances', []), feats['n_frames'], hop, sr)
	os.makedirs(os.path.dirname(out), exist_ok=True)
	np.savez(out, mel=feats['mel'], rmvpe=feats['rmvpe'], speech=labels['speech'], start=labels['start'], stop=labels['stop'])
	return out


def ensure_caches(data_dir, rmvpe=None, rmvpe_path=None, device=None, force=False):
	paths = []
	bundle = rmvpe
	for jp in list_take_jsons(data_dir):
		cp = cache_path_for(jp)
		if force or not os.path.isfile(cp):
			if bundle is None:
				print('loading RMVPE for cache…')
				bundle = load_rmvpe(rmvpe_path, device)
			print('cache', os.path.basename(jp))
			cache_take(jp, bundle)
		paths.append(cp)
	return paths


def load_npz(path):
	z = np.load(path)
	return {k: z[k] for k in ('mel', 'rmvpe', 'speech', 'start', 'stop')}


def count_params(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dark_style(root):
	from tkinter import ttk
	style = ttk.Style(root)
	try:
		style.theme_use('clam')
	except Exception:
		pass
	style.configure('Dark.TButton', background='#2a2a36', foreground='#f4f4f8', bordercolor='#3d3d4a', lightcolor='#2a2a36', darkcolor='#1a1a22', focuscolor='#2a2a36', padding=(12, 7))
	style.map('Dark.TButton', background=[('active', '#3a3a48'), ('pressed', '#1e1e28')], foreground=[('disabled', '#777')])
	style.configure('Accent.TButton', background='#ff6b9d', foreground='#14141c', bordercolor='#ff6b9d', lightcolor='#ff6b9d', darkcolor='#c44d78', focuscolor='#ff6b9d', padding=(12, 7))
	style.map('Accent.TButton', background=[('active', '#ff8fb3'), ('pressed', '#e85a88')])
	style.configure('Dark.TCombobox', fieldbackground='#1e1e28', background='#2a2a36', foreground='#f4f4f8', arrowcolor='#f4f4f8', bordercolor='#3d3d4a', lightcolor='#2a2a36', darkcolor='#1a1a22', padding=4)
	style.map('Dark.TCombobox', fieldbackground=[('readonly', '#1e1e28')], foreground=[('readonly', '#f4f4f8')])
	root.option_add('*TCombobox*Listbox.background', '#1e1e28')
	root.option_add('*TCombobox*Listbox.foreground', '#f4f4f8')
	root.option_add('*TCombobox*Listbox.selectBackground', '#ff6b9d')
	return style


def dark_button(parent, text, command, accent=False):
	from tkinter import ttk
	return ttk.Button(parent, text=text, command=command, style='Accent.TButton' if accent else 'Dark.TButton')


def plot_vad(audio, speech_p, start_p, stop_p, segs, path, sr=SAMPLE_RATE, hop=HOP_LENGTH):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	t = np.arange(len(audio)) / sr
	ft = np.arange(len(speech_p)) * hop / sr
	fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
	ax[0].plot(t, audio, color='#bbb', lw=0.6)
	for a, b in segs:
		ax[0].axvspan(a, b, color='#ff6b9d', alpha=0.25)
	ax[0].set_ylabel('wav')
	ax[1].plot(ft, speech_p, label='speech', color='#7aa2ff')
	ax[1].plot(ft, start_p, label='start', color='#5ad68a')
	ax[1].plot(ft, stop_p, label='stop', color='#ff6b9d')
	ax[1].set_ylim(-0.02, 1.05)
	ax[1].legend(loc='upper right')
	ax[1].set_xlabel('s')
	fig.tight_layout()
	fig.savefig(path, dpi=130)
	plt.close(fig)
