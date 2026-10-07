import argparse
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont
import numpy as np
import torch
from dataset import Recorder, kick_mic_permission
from models import infer_vad_coreml, load_vad, load_vad_coreml
from utils import DEFAULT_CKPT, DEFAULT_DATA, DEFAULT_OUT, DEFAULT_RUNS, SAMPLE_RATE, dark_button, dark_style, decode_segments, extract_mel, get_device, load_audio, plot_vad, resolve_ckpt, save_audio, takes_dir, to_16k, write_json


def predict(model, feats, device):
	mel = torch.from_numpy(feats['mel']).unsqueeze(0).to(device)
	with torch.no_grad():
		logits = model(mel)[0].float().cpu()
	prob = torch.sigmoid(logits).numpy()
	return prob[:, 0], prob[:, 1], prob[:, 2]


def predict_coreml(ml, feats):
	logits = infer_vad_coreml(ml, feats['mel'])
	prob = 1.0 / (1.0 + np.exp(-logits))
	return prob[:, 0], prob[:, 1], prob[:, 2]


def cut_file(audio, segs, stem, out_dir, sr=SAMPLE_RATE):
	os.makedirs(out_dir, exist_ok=True)
	paths = []
	for i, (a, b) in enumerate(segs, 1):
		x0, x1 = int(a * sr), int(b * sr)
		x1 = min(len(audio), max(x1, x0 + 1))
		name = '%s_%02d_%.2f-%.2f.wav' % (stem, i, a, b)
		path = os.path.join(out_dir, name)
		save_audio(path, audio[x0:x1], sr)
		paths.append(path)
	return paths


def analyze_array(model, audio, device, speech_th=0.45, start_th=0.30, stop_th=0.35, min_dur=0.20, coreml=None):
	audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
	t0 = time.perf_counter()
	feats = extract_mel(audio)
	if coreml is not None:
		speech_p, start_p, stop_p = predict_coreml(coreml, feats)
	else:
		speech_p, start_p, stop_p = predict(model, feats, device)
		if getattr(device, 'type', None) == 'mps':
			torch.mps.synchronize()
	segs = decode_segments(speech_p, start_p, stop_p, speech_th=speech_th, start_th=start_th, stop_th=stop_th, min_dur=min_dur)
	return {'audio': audio, 'feats': feats, 'speech': speech_p, 'start': start_p, 'stop': stop_p, 'segs': segs, 'ms': (time.perf_counter() - t0) * 1000}


def run_one(model, audio_path, device, out_dir, speech_th, start_th, stop_th, min_dur, plot, coreml=None):
	audio = load_audio(audio_path)
	res = analyze_array(model, audio, device, speech_th, start_th, stop_th, min_dur, coreml=coreml)
	stem = os.path.splitext(os.path.basename(audio_path))[0]
	print('%s  (%.0f ms, %.1fx)' % (audio_path, res['ms'], (len(audio) / SAMPLE_RATE) / max(res['ms'] / 1000, 1e-6)))
	if not res['segs']:
		print('  (no speech)')
	paths = cut_file(res['audio'], res['segs'], stem, out_dir)
	for i, ((a, b), path) in enumerate(zip(res['segs'], paths), 1):
		print('  %2d  start=%.3f  stop=%.3f  dur=%.3f  -> %s' % (i, a, b, b - a, path))
	write_json(os.path.join(out_dir, stem + '.json'), {'audio': audio_path, 'segments': [{'start': a, 'stop': b} for a, b in res['segs']], 'ms': res['ms']})
	if plot:
		plot_vad(res['audio'], res['speech'], res['start'], res['stop'], res['segs'], os.path.join(out_dir, stem + '_vad.png'))
		print('  plot', os.path.join(out_dir, stem + '_vad.png'))
	return res['segs']


class InferGui:
	def __init__(self, model_path, device, out_dir, speech_th, start_th, stop_th, min_dur, coreml_path=None):
		self.model_path = model_path
		self.coreml_path = coreml_path
		self.device = device
		self.out_dir = out_dir
		self.model = None
		self.coreml = None
		self.ready = False
		self.busy = False
		self.recording = False
		self.rec = None
		self.result = None
		self.source = ''
		self.paths = []
		self.play_t0 = None
		self.play_a = 0.0
		self.play_b = 0.0
		self.play_cursor = None
		self.paused = False
		self.pause_at = None
		self.play_gen = 0
		self.root = tk.Tk()
		self.speech_th = tk.DoubleVar(master=self.root, value=speech_th)
		self.start_th = tk.DoubleVar(master=self.root, value=start_th)
		self.stop_th = tk.DoubleVar(master=self.root, value=stop_th)
		self.min_dur = tk.DoubleVar(master=self.root, value=min_dur)
		self.root.title('LiquidVAD — infer')
		self.root.configure(bg='#14141c')
		self.root.geometry('900x820')
		self.root.minsize(760, 680)
		dark_style(self.root)
		ui = tkfont.Font(family='Helvetica', size=14)
		big = tkfont.Font(family='Helvetica', size=24, weight='bold')
		tiny = tkfont.Font(family='Helvetica', size=12)
		tk.Label(self.root, text='LiquidVAD', fg='#ff8fb3', bg='#14141c', font=big).pack(pady=(14, 2))
		self.status = tk.Label(self.root, text='loading model…', fg='#ffd27a', bg='#14141c', font=ui)
		self.status.pack()
		row = tk.Frame(self.root, bg='#14141c')
		row.pack(pady=10)
		from tkinter import ttk
		dark_button(row, 'open wav', self.open_wav).pack(side='left', padx=5)
		self.rec_btn = dark_button(row, 'record', self.toggle_rec, accent=True)
		self.rec_btn.pack(side='left', padx=5)
		dark_button(row, 'play all cuts', self.play_all).pack(side='left', padx=5)
		dark_button(row, 'save cuts', self.save_cuts).pack(side='left', padx=5)

		takes = sorted(n for n in os.listdir(takes_dir(DEFAULT_DATA)) if n.endswith('.wav')) if os.path.isdir(takes_dir(DEFAULT_DATA)) else []
		if takes:
			quick = tk.Frame(self.root, bg='#14141c')
			quick.pack()
			tk.Label(quick, text='takes', fg='#8b8b99', bg='#14141c', font=tiny).pack(side='left', padx=(0, 6))
			self.take_var = tk.StringVar(master=self.root, value=takes[0])
			cb = ttk.Combobox(quick, textvariable=self.take_var, values=takes, state='readonly', width=16, style='Dark.TCombobox')
			cb.pack(side='left')
			dark_button(quick, 'load', self.load_take).pack(side='left', padx=6)
		else:
			self.take_var = None

		self.wave = tk.Canvas(self.root, height=168, bg='#1e1e28', highlightthickness=0)
		self.wave.pack(fill='x', padx=24, pady=(12, 4))
		self.wave.bind('<Button-1>', self._seek_click)
		self.wave.bind('<Configure>', lambda e: self._draw_wave())
		self.info = tk.Label(self.root, text='', fg='#c8c8d4', bg='#14141c', font=tiny)
		self.info.pack()
		sl = tk.Frame(self.root, bg='#14141c')
		sl.pack(fill='x', padx=28, pady=8)
		self._slider(sl, 'speech', self.speech_th, 0.10, 0.90)
		self._slider(sl, 'start', self.start_th, 0.10, 0.90)
		self._slider(sl, 'stop', self.stop_th, 0.10, 0.90)
		self._slider(sl, 'min s', self.min_dur, 0.05, 1.50)
		self.listbox = tk.Listbox(self.root, bg='#1e1e28', fg='#f4f4f8', selectbackground='#ff6b9d', font=tiny, bd=0, highlightthickness=0, height=10)
		self.listbox.pack(fill='both', expand=True, padx=24, pady=(8, 8))
		self.listbox.bind('<Double-Button-1>', lambda e: self.play_selected())
		self.listbox.bind('<Return>', lambda e: self.play_selected())
		bot = tk.Frame(self.root, bg='#14141c')
		bot.pack(pady=(0, 16))
		self.pause_btn = dark_button(bot, 'pause', self.toggle_pause)
		self.pause_btn.pack(side='left', padx=5)
		dark_button(bot, 'play selected', self.play_selected).pack(side='left', padx=5)
		dark_button(bot, 'play full', self.play_full).pack(side='left', padx=5)
		dark_button(bot, 'stop audio', self.stop_audio).pack(side='left', padx=5)
		self.root.bind('<space>', lambda e: self.toggle_pause() if (self.play_t0 is not None or self.paused) else self.play_selected())
		self.root.protocol('WM_DELETE_WINDOW', self._quit)
		self._tick()
		threading.Thread(target=self._load_models, daemon=True).start()

	def _slider(self, parent, name, var, lo, hi):
		box = tk.Frame(parent, bg='#14141c')
		box.pack(side='left', expand=True, fill='x', padx=6)
		tk.Label(box, text=name, fg='#8b8b99', bg='#14141c').pack()
		sc = tk.Scale(box, from_=lo, to=hi, resolution=0.01, orient='horizontal', variable=var, bg='#14141c', fg='#f4f4f8', troughcolor='#2a2a36', highlightthickness=0, command=lambda _: self._redecode())
		sc.pack(fill='x')

	def _set(self, text, color='#ffd27a'):
		self.status.config(text=text, fg=color)

	def _load_models(self):
		try:
			self._set('loading VAD…')
			self.model, ckpt = load_vad(self.model_path, self.device)
			if self.coreml_path and os.path.exists(self.coreml_path):
				self._set('loading CoreML (ANE)…')
				self.coreml = load_vad_coreml(self.coreml_path)
			self.ready = True
			kind = 'coreml-ane' if self.coreml is not None else 'pytorch'
			self.root.after(0, lambda: self._set('ready (%s) — open a wav, load a take, or record  (epoch %s, val %.3f)' % (kind, ckpt.get('epoch'), ckpt.get('loss') or 0), '#7cffb2'))
		except Exception as e:
			self.root.after(0, lambda: self._set('load failed: %s' % e, '#ff8b8b'))

	def load_take(self):
		if not self.take_var:
			return
		path = os.path.join(takes_dir(DEFAULT_DATA), self.take_var.get())
		self._run_path(path)

	def open_wav(self):
		path = filedialog.askopenfilename(title='Open wav', filetypes=[('WAV', '*.wav'), ('All', '*.*')])
		if path:
			self._run_path(path)

	def toggle_rec(self):
		if self.recording:
			self._stop_rec()
			return
		if not self.ready or self.busy:
			return
		kick_mic_permission()
		self.stop_audio()
		self.rec = Recorder()
		try:
			self.rec.start()
		except Exception as e:
			self._set('mic failed: %s' % e, '#ff8b8b')
			return
		self.recording = True
		self.rec_btn.config(text='stop rec')
		self._set('recording — talk, then Stop rec. I will cut automatically.', '#ff8fb3')

	def _stop_rec(self):
		if not self.rec:
			return
		raw, _ = self.rec.samples()
		sr = self.rec.sr
		self.rec.stop()
		self.rec = None
		self.recording = False
		self.rec_btn.config(text='record')
		audio = to_16k(raw, sr)
		if float(np.max(np.abs(audio))) < 1e-4:
			self._set('recording was silent', '#ff8b8b')
			return
		self.source = 'mic'
		self._run_audio(audio, 'mic')

	def _run_path(self, path):
		if not self.ready or self.busy:
			self._set('wait — model still loading' if not self.ready else 'already running', '#ffd27a')
			return
		self.source = path
		self._set('loading %s…' % os.path.basename(path))
		try:
			audio = load_audio(path)
		except Exception as e:
			self._set('read failed: %s' % e, '#ff8b8b')
			return
		self._run_audio(audio, os.path.splitext(os.path.basename(path))[0])

	def _run_audio(self, audio, stem):
		if not self.ready or self.busy:
			return
		self.busy = True
		self.stem = stem
		self._set('cutting…', '#ffd27a')

		def work():
			try:
				res = analyze_array(self.model, audio, self.device, self.speech_th.get(), self.start_th.get(), self.stop_th.get(), self.min_dur.get(), coreml=self.coreml)
				err = None
			except Exception as e:
				res, err = None, e
			self.root.after(0, lambda: self._done(res, err))

		threading.Thread(target=work, daemon=True).start()

	def _done(self, res, err):
		self.busy = False
		if err:
			self._set('infer failed: %s' % err, '#ff8b8b')
			return
		self.result = res
		self.paths = cut_file(res['audio'], res['segs'], getattr(self, 'stem', 'cut'), self.out_dir)
		self._refresh_list()
		self._draw_wave()
		n = len(res['segs'])
		rt = (len(res['audio']) / SAMPLE_RATE) / max(res['ms'] / 1000, 1e-6)
		self._set('%d cut%s  ·  %.0f ms  ·  %.1fx  ·  saved to out/' % (n, '' if n == 1 else 's', res['ms'], rt), '#7cffb2')
		print('%s  %d cuts  %.0f ms' % (self.source, n, res['ms']))
		for i, ((a, b), path) in enumerate(zip(res['segs'], self.paths), 1):
			print('  %2d  %.3f–%.3f  %s' % (i, a, b, path))
		if n:
			self.listbox.selection_set(0)

	def _redecode(self):
		if not self.result or self.busy:
			return
		self.result['segs'] = decode_segments(self.result['speech'], self.result['start'], self.result['stop'], speech_th=self.speech_th.get(), start_th=self.start_th.get(), stop_th=self.stop_th.get(), min_dur=self.min_dur.get())
		self.paths = []
		self._refresh_list()
		self._draw_wave()
		self.info.config(text='%d cuts (thresholds only — not re-saved until Save cuts)' % len(self.result['segs']))

	def _refresh_list(self):
		self.listbox.delete(0, 'end')
		if not self.result:
			return
		for i, (a, b) in enumerate(self.result['segs'], 1):
			self.listbox.insert('end', '%2d   %6.2f → %6.2f    (%.2fs)' % (i, a, b, b - a))
		self.info.config(text='%d cuts' % len(self.result['segs']))

	def _draw_wave(self):
		c = self.wave
		c.delete('all')
		w, h = max(c.winfo_width(), 20), max(c.winfo_height(), 20)
		c.create_rectangle(0, 0, w, h, fill='#1e1e28', width=0)
		if not self.result:
			return
		audio = self.result['audio']
		dur = max(len(audio) / SAMPLE_RATE, 1e-6)
		for a, b in self.result['segs']:
			x0, x1 = a / dur * w, b / dur * w
			c.create_rectangle(x0, 0, x1, h, fill='#3a2030', outline='')
		step = max(1, len(audio) // max(w, 1))
		mid = h * 0.42
		amp = h * 0.36
		pts = []
		for x in range(w):
			i = min(len(audio) - 1, x * step)
			pts.extend((x, mid - float(audio[i]) * amp))
		if len(pts) >= 4:
			c.create_line(*pts, fill='#d0d0dc', width=1)
		sp = self.result['speech']
		if sp.size:
			hop = dur / max(len(sp), 1)
			curve = []
			for i, v in enumerate(sp):
				curve.extend((i * hop / dur * w, h - 8 - float(v) * (h * 0.22)))
			if len(curve) >= 4:
				c.create_line(*curve, fill='#7aa2ff', width=1)
		if self.play_cursor is not None:
			x = self.play_cursor / dur * w
			c.create_line(x, 0, x, h, fill='#ff6b9d', width=2)

	def _seek_click(self, ev):
		if not self.result:
			return
		dur = len(self.result['audio']) / SAMPLE_RATE
		t = max(0.0, min(dur, ev.x / max(self.wave.winfo_width(), 1) * dur))
		for a, b in self.result['segs']:
			if a <= t <= b:
				self._play_range(a, b)
				return
		self._play_range(t, min(dur, t + 2.0))

	def _play_range(self, a, b):
		if not self.result:
			return
		import sounddevice as sd
		audio = self.result['audio']
		x0, x1 = int(a * SAMPLE_RATE), int(b * SAMPLE_RATE)
		x1 = min(len(audio), max(x1, x0 + 1))
		sd.stop()
		sd.play(audio[x0:x1], SAMPLE_RATE)
		self.play_t0 = time.perf_counter()
		self.play_a, self.play_b = a, b
		self.play_cursor = a
		self.paused = False
		self.pause_at = None
		self.pause_btn.config(text='pause')

	def play_selected(self):
		if not self.result or not self.result['segs']:
			return
		self.play_gen += 1
		sel = self.listbox.curselection()
		i = int(sel[0]) if sel else 0
		a, b = self.result['segs'][i]
		self.listbox.selection_clear(0, 'end')
		self.listbox.selection_set(i)
		self._play_range(a, b)

	def play_full(self):
		if not self.result:
			return
		self.play_gen += 1
		self._play_range(0.0, len(self.result['audio']) / SAMPLE_RATE)

	def play_all(self):
		if not self.result or not self.result['segs']:
			return
		self.play_gen += 1
		self._play_chain(0, self.play_gen)

	def _play_chain(self, i, gen):
		if gen != self.play_gen or not self.result or i >= len(self.result['segs']):
			return
		a, b = self.result['segs'][i]
		self.listbox.selection_clear(0, 'end')
		self.listbox.selection_set(i)
		self.listbox.see(i)
		self._play_range(a, b)
		self.root.after(int((b - a) * 1000 + 220), lambda: self._play_chain(i + 1, gen))

	def toggle_pause(self):
		if self.play_t0 is not None:
			try:
				import sounddevice as sd
				sd.stop()
			except Exception:
				pass
			t = min(self.play_b, self.play_a + (time.perf_counter() - self.play_t0))
			self.play_t0 = None
			self.paused = True
			self.pause_at = t
			self.play_cursor = t
			self.play_gen += 1
			self.pause_btn.config(text='resume')
			self._draw_wave()
			return
		if self.paused and self.pause_at is not None and self.result:
			end = self.play_b if self.play_b > self.pause_at else len(self.result['audio']) / SAMPLE_RATE
			self._play_range(self.pause_at, end)
			return
		self.play_selected()

	def stop_audio(self):
		self.play_gen += 1
		try:
			import sounddevice as sd
			sd.stop()
		except Exception:
			pass
		self.play_t0 = None
		self.play_cursor = None
		self.paused = False
		self.pause_at = None
		self.pause_btn.config(text='pause')
		self._draw_wave()

	def save_cuts(self):
		if not self.result:
			return
		stem = getattr(self, 'stem', 'cut')
		self.paths = cut_file(self.result['audio'], self.result['segs'], stem, self.out_dir)
		write_json(os.path.join(self.out_dir, stem + '.json'), {'audio': self.source, 'segments': [{'start': a, 'stop': b} for a, b in self.result['segs']]})
		self._set('saved %d cuts → %s' % (len(self.paths), self.out_dir), '#7cffb2')

	def _tick(self):
		if self.recording and self.rec is not None:
			t = self.rec.n_samples() / float(self.rec.sr)
			self.info.config(text='recording  %.1fs   peak %.3f' % (t, self.rec.peak))
		if self.play_t0 is not None:
			t = self.play_a + (time.perf_counter() - self.play_t0)
			if t >= self.play_b:
				self.play_t0 = None
				self.play_cursor = None
			else:
				self.play_cursor = t
			self._draw_wave()
		self.root.after(40, self._tick)

	def _quit(self):
		self.stop_audio()
		if self.rec:
			self.rec.stop()
		self.root.destroy()

	def run(self):
		self.root.mainloop()
		self.stop_audio()


def record_mic(seconds=None):
	import sounddevice as sd
	print('mic on — press Enter to cut' if seconds is None else 'mic on — recording %.1fs' % seconds)
	chunks = []

	def cb(indata, frames, t, status):
		if status:
			print(status)
		chunks.append(indata.copy().reshape(-1))

	with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=512, callback=cb):
		if seconds is None:
			try:
				input()
			except EOFError:
				time.sleep(3)
		else:
			time.sleep(seconds)
	if not chunks:
		raise SystemExit('no mic audio')
	return np.clip(np.concatenate(chunks), -1.0, 1.0).astype(np.float32)


def main():
	p = argparse.ArgumentParser(description='LiquidVAD infer — GUI by default, or CLI with --audio / --mic')
	p.add_argument('--model', default=DEFAULT_CKPT)
	p.add_argument('--audio', default=None, help='wav or a folder of wavs (CLI)')
	p.add_argument('--mic', action='store_true', help='CLI record then cut')
	p.add_argument('--gui', action='store_true', help='open the playback / auto-cut window')
	p.add_argument('--seconds', type=float, default=None)
	p.add_argument('--out', default=DEFAULT_OUT)
	p.add_argument('--coreml', default=os.path.join(DEFAULT_RUNS, 'liquid_vad.mlpackage'))
	p.add_argument('--pt', action='store_true', help='force PyTorch even if CoreML exists')
	p.add_argument('--device', default='cpu', help='PyTorch device if not using CoreML')
	p.add_argument('--speech-th', type=float, default=0.45)
	p.add_argument('--start-th', type=float, default=0.30)
	p.add_argument('--stop-th', type=float, default=0.35)
	p.add_argument('--min-dur', type=float, default=0.20)
	p.add_argument('--plot', action='store_true')
	args = p.parse_args()

	args.model = resolve_ckpt(args.model)
	if not os.path.isfile(args.model):
		raise SystemExit('no VAD checkpoint at %s — run train.py' % args.model)

	use_gui = args.gui or (not args.audio and not args.mic)
	device = get_device(args.device)
	os.makedirs(args.out, exist_ok=True)
	legacy_ml = os.path.join(DEFAULT_RUNS, 'yuna_vad.mlpackage')
	default_ml = os.path.join(DEFAULT_RUNS, 'liquid_vad.mlpackage')
	if not args.pt and not os.path.exists(args.coreml) and os.path.exists(legacy_ml) and os.path.abspath(args.coreml) == os.path.abspath(default_ml):
		args.coreml = legacy_ml

	coreml = None if args.pt or not os.path.exists(args.coreml) else load_vad_coreml(args.coreml)
	if use_gui:
		InferGui(args.model, device, args.out, args.speech_th, args.start_th, args.stop_th, args.min_dur, None if args.pt else args.coreml).run()
		return

	print('device:', 'coreml-ane' if coreml is not None else device)
	print('loading VAD', args.model)
	model, ckpt = load_vad(args.model, device)
	print('epoch', ckpt.get('epoch'), 'loss', ckpt.get('loss'))

	if args.mic:
		audio = record_mic(args.seconds)
		tmp = os.path.join(args.out, 'mic.wav')
		save_audio(tmp, audio)
		print('saved', tmp)
		run_one(model, tmp, device, args.out, args.speech_th, args.start_th, args.stop_th, args.min_dur, args.plot, coreml=coreml)
		return

	path = args.audio
	files = sorted(os.path.join(path, n) for n in os.listdir(path) if n.lower().endswith('.wav')) if os.path.isdir(path) else [path]
	if not files:
		raise SystemExit('no wavs at %s' % path)
	for f in files:
		run_one(model, f, device, args.out, args.speech_th, args.start_th, args.stop_th, args.min_dur, args.plot, coreml=coreml)


if __name__ == '__main__':
	main()
