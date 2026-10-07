import argparse
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import (DEFAULT_DATA, F0_MAX, F0_MIN, HOP_LENGTH, NOISE_WAV, SAMPLE_RATE, cache_take, dark_button, dark_style, extract_mel, get_device, list_take_jsons, load_audio, load_npz, load_rmvpe, read_json, save_audio, shown_dir, takes_dir, to_16k, wav_is_silent, write_json, )

PER_TAKE = 5  # one list; loud/quiet/near/far done in-room on purpose; 20 takes × 5 = 100 clips, each kept clean and with pitch + ambience
CLIPS = ['Hey Yuna, can you hear me okay?', "It's me. Just checking the mic.", "Alright, let's start.", 'I am right here at the desk.', 'Still me. Same sentence, new breath.', 'What time is it right now?', 'Can you remind me what we were talking about yesterday?', 'Do you want to hear something kind of random?', 'I will say this once, then stop.', 'Okay. That was the whole question.', 'I got home a bit late, but I wanted to say good night before I sleep.', 'Tomorrow I have a lot to do, so I might be quiet until the evening.', 'If I forget, can you poke me around six?', 'The day ran long, and I am still talking.', 'Good night for now.', 'Wait. No, that is not what I meant.', 'I was thinking… maybe we try again from the start.', 'Yeah. That sounds better.', 'Hold on.', 'There. That is the end of the thought.', 'Play something soft, not too loud.', 'Set a timer for ten minutes, then remind me to stretch.', 'And dim the lights a little, please.', 'Actually, leave the lights. I changed my mind.', 'Thank you.', "I'm tired, but I still wanted to talk to you for a bit.", 'You always make the day feel less heavy, you know?', 'Okay, your turn. Say something cute.', 'I will keep going after this pause, and the cut is only at the end.', 'Alright. Your turn starts when I stop.', 'My name is Yuki, and I live with too many unfinished projects.', 'Repeat after me: one, two, three, four, five.', 'Six, seven, eight, nine, ten.', 'That was just counting.', 'We can stop the counting there.', 'Yuna, 聞こえてる？ちょっとテストするね。', '今日は短く話すから、終わりが来たらそこで切って。', 'Okay, back to English. That was the Japanese one.', 'もう一回。ゆっくり言うね。', 'はい、ここまで。', 'So anyway, I was walking back and I just… I forgot what I was going to say.', 'Never mind. It was not important.', 'We can pick this up later if you want.', 'I took a breath in the middle on purpose. The turn is still going.', 'Only the ending is the stop.', 'This one is longer, so stay with me through the little pauses, okay?', 'I will breathe, then keep going, and only when I am actually done I want the stop.', 'That is the whole point of this. End it here.', 'Short.', 'Even shorter.', 'Good night. Let your thoughts drift away.', 'Sometimes I just like hearing your voice.', "That's it, you're doing fine. Listen carefully.", 'Good morning. Did you have a nice dream?', 'The room is quiet tonight.', 'Can you open the window a crack?', 'I left my keys by the door again.', "Don't let me forget the laundry.", "I'll be back in a minute.", "I'm back.", 'Tell me something about the weather, even if you have to guess.', "I don't need a long answer.", 'Just a yes or a no is fine.', 'Yes.', 'No, wait, yes.', "I'm making tea.", 'The kettle is going, and I am still talking over it.', 'Still me.', 'Pour, pause, then the rest of the sentence.', "Tea's done.", "Let's try a list. Apples.", 'Oranges.', 'Bread, and then nothing else.', 'I said bread, then I stopped.', 'New sentence after the stop.', 'If I trail off like this…', '…I might still be in the same turn.', 'But this one is a new turn.', 'And this one ends cleanly.', 'Good.', 'Yuna, stay with me while I look for the charger.', 'Found it.', 'Plugged in.', 'You can relax.', "That's all for this little search.", 'Hello from the desk.', 'Hello again, a step farther from the mic.', 'Same words from the doorway: hello.', 'Back at the desk. One more hello.', 'And that hello is finished.', 'Count the pauses, not the breaths.', 'One complete thought, with a pause in the center, still one clip.', 'Second thought.', 'Third.', 'Fourth, and I mean it this time.', 'Last group.', 'Thank you for listening all the way through.', 'I will speak the way I actually speak.', 'This line can be close, far, quiet, or loud. I choose that in the room.', 'This is clip one hundred. End it here.', ]
assert len(CLIPS) == 100, len(CLIPS)
TAKES = [CLIPS[i:i + PER_TAKE] for i in range(0, len(CLIPS), PER_TAKE)]
N_CLIPS = len(CLIPS)
_NOISE = None


def prompts_for(take_i):
	return TAKES[(int(take_i) - 1) % len(TAKES)]


def load_ambience(path=None):
	global _NOISE
	path = path or NOISE_WAV
	if _NOISE is None:
		if not os.path.isfile(path):
			raise FileNotFoundError('ambience missing at %s' % path)
		import soundfile as sf
		audio, sr = sf.read(path, always_2d=False, dtype='float32')
		if getattr(audio, 'ndim', 1) > 1:
			audio = audio.mean(axis=-1)
		audio = to_16k(np.asarray(audio, dtype=np.float32).reshape(-1), sr)
		_NOISE = np.ascontiguousarray(audio, dtype=np.float32)
	return _NOISE


def mix_ambience(audio, noise, rng, snr_mean=10.0, snr_std=6.0):
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	noise = np.asarray(noise, dtype=np.float32).reshape(-1)
	n = int(audio.size)
	if n == 0 or noise.size == 0:
		return audio
	if noise.size < n:
		noise = np.tile(noise, int(math.ceil(n / float(noise.size))))
	start = int(rng.randint(0, noise.size - n + 1))
	bed = noise[start:start + n]
	snr = float(np.clip(rng.normal(snr_mean, snr_std), -3.0, 25.0))
	a_rms = float(np.sqrt(np.mean(audio * audio) + 1e-8))
	b_rms = float(np.sqrt(np.mean(bed * bed) + 1e-8))
	gain = a_rms / (b_rms * (10.0**(snr / 20.0)))
	mixed = audio + bed * np.float32(gain)
	peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
	if peak > 1.0:
		mixed = mixed / peak
	return mixed.astype(np.float32)


def pitch_and_noise(audio, rng, noise=None):
	"""Gaussian semitone shift, then a slice of data/noise.wav. Returns audio, semitones."""
	audio = np.asarray(audio, dtype=np.float32).reshape(-1)
	semis = float(np.clip(rng.normal(0.0, 1.5), -5.0, 5.0))
	out = audio
	if audio.size >= 2048 and abs(semis) >= 0.05:
		try:
			import librosa
			out = np.asarray(librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semis), dtype=np.float32)
		except Exception:
			semis = 0.0
			out = audio
	else:
		semis = 0.0
	if noise is None:
		noise = load_ambience()
	return mix_ambience(out, noise, rng), semis


def gaussian_pitch_wobble(n, rng, sigma_frames=8.0, amp=0.03):
	n = int(n)
	if n <= 0:
		return np.zeros(0, dtype=np.float32)
	raw = rng.normal(0.0, 1.0, size=n).astype(np.float32)
	half = max(1, int(3 * sigma_frames))
	t = np.arange(-half, half + 1, dtype=np.float32)
	ker = np.exp(-0.5 * (t / float(sigma_frames))**2)
	ker /= ker.sum()
	sm = np.convolve(raw, ker, mode='same').astype(np.float32)
	sm = sm / (float(np.std(sm)) + 1e-6) * float(amp)
	return sm.astype(np.float32)


def shift_pitch_line(rmvpe, semitones, rng):
	"""Move the log-f0 line by the same semitones, plus a smooth gaussian wobble."""
	out = np.array(rmvpe, dtype=np.float32, copy=True)
	span = math.log(F0_MAX) - math.log(F0_MIN)
	delta = float(semitones) * (math.log(2.0) / 12.0) / span
	wobble = gaussian_pitch_wobble(out.shape[-1], rng)
	voiced = out[1] >= 0.5
	out[0, voiced] = np.clip(out[0, voiced] + np.float32(delta) + wobble[voiced], 0.0, 1.0)
	return out


def fit_time(mel, n):
	mel = np.asarray(mel, dtype=np.float32)
	t = int(mel.shape[-1])
	n = int(n)
	if t == n:
		return mel
	if t > n:
		left = (t - n) // 2
		return mel[..., left:left + n].copy()
	pad = n - t
	return np.pad(mel, [(0, 0)] * (mel.ndim - 1) + [(0, pad)]).astype(np.float32)


def wav_for_cache(cache_path):
	data_dir = os.path.dirname(os.path.dirname(os.path.abspath(cache_path)))
	stem = os.path.splitext(os.path.basename(cache_path))[0]
	return os.path.join(takes_dir(data_dir), stem + '.wav')


def pair_paths(data_dir, take_i, clip_i):
	folder = shown_dir(data_dir)
	stem = 'take_%02d_%02d' % (int(take_i), int(clip_i))
	return os.path.join(folder, stem + '_clean.wav'), os.path.join(folder, stem + '_noise.wav')


def write_pair(data_dir, take_i, clip_i, clip, rng=None):
	"""Save one clip twice: the take as marked, and the same clip with pitch + ambience."""
	rng = rng or np.random.RandomState()
	clean, noisy = pair_paths(data_dir, take_i, clip_i)
	clip = np.asarray(clip, dtype=np.float32).reshape(-1)
	mixed, semis = pitch_and_noise(clip, rng)
	save_audio(clean, clip)
	save_audio(noisy, mixed)
	return clean, noisy, semis


def show_saved(data_dir=None):
	data_dir = data_dir or DEFAULT_DATA
	n = 0
	for jp in list_take_jsons(data_dir):
		meta = read_json(jp)
		wav = meta.get('wav') or os.path.join(os.path.dirname(jp), os.path.basename(meta.get('path', '')))
		if not os.path.isfile(wav):
			wav = os.path.join(os.path.dirname(jp), os.path.basename(meta.get('path', '')))
		if not os.path.isfile(wav):
			print('missing wav for', jp)
			continue
		audio = load_audio(wav)
		sr = SAMPLE_RATE
		take_i = int(meta.get('take') or 0)
		for i, u in enumerate(meta.get('utterances') or [], 1):
			a = int(max(0.0, float(u['start'])) * sr)
			b = int(max(float(u['start']), float(u['stop'])) * sr)
			b = min(len(audio), max(b, a + 1))
			clean, noisy, semis = write_pair(data_dir, take_i, i, audio[a:b], np.random.RandomState(take_i * 100 + i))
			print('clean ', clean)
			print('noise ', noisy, '  pitch %+.2f st' % semis)
			n += 1
	print('shown %d / %d clips, each as clean and noise' % (n, N_CLIPS))
	return n


class LiquidVADSet(Dataset):
	"""Each index is one take window as (clean, noisy); noisy = same frames after gaussian pitch shift, f0 wobble, and a random slice of data/noise.wav."""
	def __init__(self, cache_paths, augment=False, crop_sec=6.0, hop=HOP_LENGTH, sr=SAMPLE_RATE):
		self.augment = augment
		self.crop = int(crop_sec * sr / hop) if crop_sec else 0
		self.hop = hop
		self.sr = sr
		self._audio_cache = {}
		self.items = []
		for path in cache_paths:
			z = load_npz(path)
			item = {k: np.ascontiguousarray(z[k]) for k in z}
			item['_wav'] = wav_for_cache(path)
			item['_a'] = 0
			item['_b'] = int(item['mel'].shape[-1])
			self.items.append(item)

	def __len__(self):
		return len(self.items)

	def _audio(self, path):
		if path not in self._audio_cache:
			self._audio_cache[path] = load_audio(path)
		return self._audio_cache[path]

	def _slice(self, item, a, b):
		out = {}
		for k, v in item.items():
			if k in ('_wav', '_a', '_b'):
				continue
			out[k] = v[..., a:b].copy()
		out['_wav'] = item['_wav']
		out['_a'] = int(item['_a']) + int(a)
		out['_b'] = int(item['_a']) + int(b)
		return out

	def _crop(self, item, rng):
		t = item['mel'].shape[-1]
		if not self.crop or t <= self.crop:
			return item
		stop = item['stop']
		peaks = np.where(stop >= 0.5)[0]
		lo = max(1, self.crop // 4)
		hi = max(lo + 1, self.crop - 8)
		starts = item['start']
		spk = np.where(starts >= 0.5)[0]
		r = rng.random()
		if peaks.size and r < 0.62:
			c = int(rng.choice(peaks))
			a = max(0, c - int(rng.randint(lo, hi)))
		elif spk.size and r < 0.80:
			c = int(rng.choice(spk))
			a = max(0, c - int(rng.randint(8, max(9, self.crop // 3))))
		else:
			a = int(rng.randint(0, t - self.crop + 1))
		b = min(t, a + self.crop)
		a = max(0, b - self.crop)
		return self._slice(item, a, b)

	def _view(self, item, noisy, rng):
		keys = ('mel', 'rmvpe', 'speech', 'start', 'stop')
		if not noisy:
			return {k: item[k] for k in keys}
		n = int(item['mel'].shape[-1])
		audio = self._audio(item['_wav'])
		s0 = int(item['_a'] * self.hop)
		s1 = max(s0 + 1, int(item['_b'] * self.hop))
		chunk = np.asarray(audio[s0:s1], dtype=np.float32)
		if chunk.size == 0:
			chunk = np.zeros(self.hop * 4, dtype=np.float32)
		mixed, semis = pitch_and_noise(chunk, rng)
		mel = fit_time(extract_mel(mixed)['mel'], n)
		return {'mel': mel.astype(np.float32), 'rmvpe': shift_pitch_line(item['rmvpe'], semis, rng), 'speech': item['speech'], 'start': item['start'], 'stop': item['stop'], }

	def __getitem__(self, idx):
		rng = np.random.RandomState(None if self.augment else (int(idx) + 17))
		item = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.items[idx].items()}
		if self.augment:
			item = self._crop(item, rng)
		clean = self._view(item, False, rng)
		noisy = self._view(item, True, rng if self.augment else np.random.RandomState(int(idx) + 91))

		def tens(d):
			return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}

		return tens(clean), tens(noisy)


def collate_vad(batch):
	flat = []
	for item in batch:
		if isinstance(item, (tuple, list)):
			flat.extend(item)
		else:
			flat.append(item)
	t = max(x['mel'].shape[-1] for x in flat)
	b = len(flat)
	mel = torch.zeros(b, flat[0]['mel'].shape[0], t)
	rmvpe = torch.zeros(b, flat[0]['rmvpe'].shape[0], t)
	speech = torch.zeros(b, t)
	start = torch.zeros(b, t)
	stop = torch.zeros(b, t)
	mask = torch.zeros(b, t)
	for i, x in enumerate(flat):
		n = x['mel'].shape[-1]
		mel[i, :, :n] = x['mel']
		rmvpe[i, :, :n] = x['rmvpe']
		speech[i, :n] = x['speech']
		start[i, :n] = x['start']
		stop[i, :n] = x['stop']
		mask[i, :n] = 1
	return {'mel': mel, 'rmvpe': rmvpe, 'speech': speech, 'start': start, 'stop': stop, 'mask': mask}


def _say(*parts):
	print(*parts, flush=True)


def list_mics():
	import sounddevice as sd
	devs = sd.query_devices()
	out = []
	_say('input devices:')
	for i, d in enumerate(devs):
		if d['max_input_channels'] < 1:
			continue
		out.append(i)
		_say('  %2d  %s  (sr=%g, ch=%d)' % (i, d['name'], d['default_samplerate'], d['max_input_channels']))
	return out


def resolve_mic(spec):
	import sounddevice as sd
	if spec is None or spec == '':
		idx = sd.default.device[0]
		return 0 if idx is None or int(idx) < 0 else int(idx)
	if str(spec).isdigit():
		return int(spec)
	needle = str(spec).lower()
	for i, d in enumerate(sd.query_devices()):
		if d['max_input_channels'] > 0 and needle in d['name'].lower():
			return i
	raise SystemExit('unknown --mic %r' % spec)


def kick_mic_permission():
	if sys.platform != 'darwin':
		return
	try:
		from AVFoundation import AVCaptureDevice, AVMediaTypeAudio
		AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeAudio, lambda ok: None)
		time.sleep(0.2)
	except Exception:
		pass


def open_mic_settings():
	if sys.platform != 'darwin':
		return
	os.system('open "x-apple.systemsettings:com.apple.settings.PrivacySecurity.extension?Privacy_Microphone" >/dev/null 2>&1 || open "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone" >/dev/null 2>&1')


class Recorder:
	def __init__(self, device=None, minutes=20):
		self.device = device
		self.sr = SAMPLE_RATE
		self.minutes = minutes
		self.buf = np.zeros(1, dtype=np.float32)
		self.n = 0
		self.level = 0.0
		self.peak = 0.0
		self.lock = threading.Lock()
		self.stream = None
		self.name = ''

	def _cb(self, indata, frames, t, status):
		if status:
			_say('mic status:', status)
		x = np.ascontiguousarray(indata, dtype=np.float32)
		if x.ndim > 1:
			x = x.mean(axis=1)
		else:
			x = x.reshape(-1)
		p = float(np.max(np.abs(x))) if x.size else 0.0
		with self.lock:
			i = self.n
			j = i + x.size
			if j > len(self.buf):
				return
			self.buf[i:j] = x
			self.n = j
			self.level = float(np.sqrt(np.mean(x * x) + 1e-12))
			if p > self.peak:
				self.peak = p

	def start(self):
		import sounddevice as sd
		info = sd.query_devices(self.device, 'input') if self.device is not None else sd.query_devices(kind='input')
		self.name = info.get('name') or '?'
		self.sr = int(info.get('default_samplerate') or SAMPLE_RATE)
		ch = 1 if int(info.get('max_input_channels') or 1) >= 1 else int(info.get('max_input_channels') or 1)
		self.buf = np.zeros(int(self.sr * self.minutes * 60), dtype=np.float32)
		self.n = 0
		self.peak = 0.0
		_say('mic:', self.name, 'idx', info.get('index', self.device), 'native_sr', self.sr)
		self.stream = sd.InputStream(device=info.get('index', self.device), samplerate=self.sr, channels=ch, dtype='float32', blocksize=0, latency='high', callback=self._cb)
		self.stream.start()

	def stop(self):
		if self.stream is not None:
			self.stream.stop()
			self.stream.close()
			self.stream = None

	def samples(self, a=0, b=None):
		with self.lock:
			n = self.n
			if b is None:
				b = n
			a, b = max(0, a), min(n, b)
			return self.buf[a:b].copy(), n

	def n_samples(self):
		with self.lock:
			return self.n


class App:
	def __init__(self, data_dir, takes, rmvpe_path, device, mic=None):
		self.data_dir = data_dir
		self.takes_left = takes
		self.rmvpe_path = rmvpe_path
		self.device = device
		self.rmvpe = None
		self.rmvpe_error = None
		self.mic_dead = False
		self.rec = Recorder(device=mic)
		self.take_i = 1
		self.line_i = 0
		self.origin = 0
		self.last_stop = 0.0
		self.phase = 'arm'
		self.mark_start = None
		self.utterances = []
		self.pending = 0
		self.lock = threading.Lock()
		os.makedirs(takes_dir(data_dir), exist_ok=True)
		os.makedirs(shown_dir(data_dir), exist_ok=True)
		threading.Thread(target=self._load_rmvpe, daemon=True).start()
		self.root = tk.Tk()
		self.root.title('LiquidVAD — record')
		self.root.configure(bg='#14141c')
		self.root.geometry('760x860')
		self.root.minsize(640, 740)
		dark_style(self.root)
		ui = tkfont.Font(family='Helvetica', size=15)
		big = tkfont.Font(family='Helvetica', size=28, weight='bold')
		read = tkfont.Font(family='Helvetica', size=22)
		tiny = tkfont.Font(family='Helvetica', size=12)
		self.title = tk.Label(self.root, text='LiquidVAD', fg='#ff8fb3', bg='#14141c', font=big)
		self.title.pack(pady=(16, 2))
		self.sub = tk.Label(self.root, text='', fg='#c8c8d4', bg='#14141c', font=ui)
		self.sub.pack()
		self.clock = tk.Label(self.root, text='00:00.0', fg='#f4f4f8', bg='#14141c', font=big)
		self.clock.pack(pady=(6, 2))
		self.meter_w = 420
		meter = tk.Canvas(self.root, width=self.meter_w, height=18, bg='#1e1e28', highlightthickness=0)
		meter.pack(pady=6)
		self.meter = meter
		self.bar = meter.create_rectangle(0, 0, 0, 18, fill='#7cffb2', width=0)
		self.level_lbl = tk.Label(self.root, text='peak 0.0000', fg='#8b8b99', bg='#14141c', font=tiny)
		self.level_lbl.pack()
		tk.Label(self.root, text='READ THIS', fg='#8b8b99', bg='#14141c', font=tiny).pack(pady=(10, 0))
		self.read = tk.Label(self.root, text='', fg='#f4f4f8', bg='#1e1e28', font=read, wraplength=680, justify='center', padx=22, pady=22)
		self.read.pack(fill='x', padx=28, pady=(6, 10))
		self.status = tk.Label(self.root, text='', fg='#ffd27a', bg='#14141c', font=ui, wraplength=680, justify='center')
		self.status.pack(pady=(0, 10))
		self.mark_btn = dark_button(self.root, 'Start Recording Data', self.mark_bound, accent=True)
		self.mark_btn.pack(fill='x', padx=80, pady=8, ipady=10)
		row = tk.Frame(self.root, bg='#14141c')
		row.pack(pady=(8, 2))
		for text, cmd in (('undo last', self.undo), ('skip line', self.skip_line), ('skip take', self.next_take), ('finish', self.finish)):
			dark_button(row, text, cmd).pack(side='left', padx=4)
		hear = tk.Frame(self.root, bg='#14141c')
		hear.pack(pady=(2, 8))
		dark_button(hear, 'hear clean', lambda: self.hear('clean')).pack(side='left', padx=4)
		dark_button(hear, 'hear noise', lambda: self.hear('noise')).pack(side='left', padx=4)
		self.listbox = tk.Listbox(self.root, bg='#1e1e28', fg='#f4f4f8', selectbackground='#ff6b9d', font=tiny, bd=0, highlightthickness=0, height=8)
		self.listbox.pack(fill='both', expand=True, padx=28, pady=(12, 18))
		self.listbox.bind('<Double-Button-1>', lambda e: self.hear('clean'))
		self.mark_btn.bind('<space>', lambda e: 'break')
		self.mark_btn.bind('<Return>', lambda e: 'break')
		self.root.bind('<space>', lambda e: self.mark_bound())
		self.root.bind('<Return>', lambda e: self.mark_bound())
		self.root.bind('<BackSpace>', lambda e: self.undo())
		self.root.bind('n', lambda e: self.skip_line())
		self.root.bind('N', lambda e: self.skip_line())
		self.root.protocol('WM_DELETE_WINDOW', self.finish)
		self._tick()

	def lines(self):
		return prompts_for(self.take_i)

	def current_text(self):
		lines = self.lines()
		if self.line_i >= len(lines):
			return ''
		return lines[self.line_i]

	def last_take(self):
		return self.take_i + self.takes_left - 1

	def clip_number(self):
		return (self.take_i - 1) * PER_TAKE + self.line_i + 1

	def take_audio(self):
		audio, n = self.rec.samples(self.origin)
		return audio, n

	def now_sec(self):
		return max(0.0, (self.rec.n_samples() - self.origin) / float(self.rec.sr))

	def _set_status(self, text, color='#ffd27a'):
		self.status.config(text=text, fg=color)

	def refresh(self):
		n_lines = len(self.lines())
		self.sub.config(text='clip %d / %d    ·    take %d / %d    ·    line %d / %d    ·    %d saved' % (min(self.clip_number(), N_CLIPS), N_CLIPS, self.take_i, self.last_take(), min(self.line_i + 1, n_lines), n_lines, len(self.utterances)))
		if self.phase == 'arm':
			self.read.config(text='One press starts the mic. It stays open until every clip in this sitting is done.')
		else:
			self.read.config(text=self.current_text() or '(take done)')
		self.listbox.delete(0, 'end')
		for i, u in enumerate(self.utterances, 1):
			bit = u.get('text') or ''
			if len(bit) > 42:
				bit = bit[:39] + '…'
			self.listbox.insert('end', '%2d  %5.2f → %5.2f   clean + noise   %s' % (i, u['start'], u['stop'], bit))
		self._sync_mark_btn()
		if self.phase == 'arm':
			self._set_status('Start Recording Data once. Then each line is Mark Start, then Mark End.', '#ffd27a')
		elif self.phase == 'mark_start':
			self._set_status('Mark Start where this line begins.', '#ffd27a')
		else:
			self._set_status('Started at %.2fs — Mark End where this line ends.' % self.mark_start, '#7cffb2')

	def _sync_mark_btn(self):
		if self.phase == 'arm':
			self.mark_btn.configure(text='Start Recording Data')
		elif self.phase == 'mark_stop':
			self.mark_btn.configure(text='Mark End')
		else:
			self.mark_btn.configure(text='Mark Start')

	def _announce(self):
		text = self.current_text()
		n_lines = len(self.lines())
		_say('')
		_say('=' * 64)
		_say('CLIP %d/%d   TAKE %d/%d   line %d/%d' % (self.clip_number(), N_CLIPS, self.take_i, self.last_take(), self.line_i + 1, n_lines))
		_say('READ:  %s' % text)
		if self.phase == 'arm':
			_say('       Button = Start Recording Data (once). The mic stays open.')
		elif self.phase == 'mark_start':
			_say('       Button = Mark Start.')
		else:
			_say('       Button = Mark End.')
		_say('=' * 64)

	def _front(self):
		self.root.deiconify()
		self.root.lift()
		self.root.focus_force()
		try:
			self.root.attributes('-topmost', True)
			self.root.after(1200, lambda: self.root.attributes('-topmost', False))
		except tk.TclError:
			pass
		if sys.platform == 'darwin':
			os.system('/usr/bin/osascript -e \'tell application "System Events" to set frontmost of the first process whose unix id is %d to true\' >/dev/null 2>&1' % os.getpid())

	def _load_rmvpe(self):
		try:
			_say('loading RMVPE in the background…')
			self.rmvpe = load_rmvpe(self.rmvpe_path, self.device)
			_say('RMVPE ready:', self.rmvpe['kind'])
		except Exception as e:
			self.rmvpe_error = e
			_say('RMVPE load failed (takes still save):', e)

	def _clip_audio(self, start, stop):
		audio, _ = self.take_audio()
		audio = to_16k(audio, self.rec.sr)
		a = int(float(start) * SAMPLE_RATE)
		b = int(float(stop) * SAMPLE_RATE)
		b = min(len(audio), max(b, a + 1))
		a = max(0, min(a, b - 1))
		return audio[a:b]

	def _write_marked(self, index):
		u = self.utterances[index - 1]
		clip = self._clip_audio(u['start'], u['stop'])
		clean, noisy, semis = write_pair(self.data_dir, self.take_i, index, clip)
		_say('  clean ', clean)
		_say('  noise ', noisy, '  pitch %+.2f st' % semis)

	def _drop_pair(self, index):
		for path in pair_paths(self.data_dir, self.take_i, index):
			if os.path.isfile(path):
				os.remove(path)

	def hear(self, kind):
		if not self.utterances:
			self._set_status('Mark a line first. Then you can hear it clean and with noise.', '#ffd27a')
			return
		sel = self.listbox.curselection()
		i = int(sel[0]) + 1 if sel else len(self.utterances)
		clean, noisy = pair_paths(self.data_dir, self.take_i, i)
		path = noisy if kind == 'noise' else clean
		if not os.path.isfile(path):
			try:
				self._write_marked(i)
			except Exception as e:
				self._set_status('preview failed: %s' % e, '#ff8b8b')
				return
			clean, noisy = pair_paths(self.data_dir, self.take_i, i)
			path = noisy if kind == 'noise' else clean
		try:
			import sounddevice as sd
			import soundfile as sf
			audio, sr = sf.read(path, always_2d=False, dtype='float32')
			sd.stop()
			sd.play(np.asarray(audio, dtype=np.float32), int(sr))
		except Exception as e:
			self._set_status('play failed: %s' % e, '#ff8b8b')
			return
		self._set_status('playing %s  ·  %s' % (kind, os.path.basename(path)), '#7cffb2' if kind == 'clean' else '#ff8fb3')

	def _arm_recording(self):
		if self.rec.stream is not None:
			self.phase = 'mark_start'
			self.refresh()
			self._announce()
			return
		self._start_mic()
		if self.rec.stream is None:
			return
		self.origin = self.rec.n_samples()
		self.last_stop = 0.0
		self.mark_start = None
		self.phase = 'mark_start'
		_say('mic is open for this sitting. Mark Start, then Mark End, on each line.')
		self.refresh()
		self._announce()

	def mark_bound(self):
		if self.phase == 'arm':
			self._arm_recording()
			return
		if not self.current_text():
			return
		t = self.now_sec()
		if self.phase == 'mark_start':
			if t < self.last_stop:
				t = self.last_stop
			self.mark_start = t
			self.phase = 'mark_stop'
			_say('  >> Mark Start  %.2fs    now Mark End where the line ends' % t)
			self.refresh()
			return
		start = self.mark_start
		if start is None:
			self.phase = 'mark_start'
			return
		if t - start < 0.12:
			_say('  !! too short after Mark Start. Mark End a little later.')
			self._set_status('Too short — Mark End a little later.', '#ff8b8b')
			return
		audio, _ = self.take_audio()
		i0, i1 = int(start * self.rec.sr), int(t * self.rec.sr)
		clip = audio[max(0, i0):max(i0 + 1, i1)]
		peak = float(np.max(np.abs(clip))) if clip.size else 0.0
		if peak < 1e-4:
			_say('  !! MIC SILENT (peak %.6f). This line was NOT saved.' % peak)
			self._set_status('MIC SILENT — enable Microphone, then mark again.', '#ff8b8b')
			self.mic_dead = True
			return
		text = self.current_text()
		u = {'start': round(float(start), 4), 'stop': round(float(t), 4), 'start_src': 'user', 'text': text}
		self.utterances.append(u)
		self.last_stop = t
		self.mark_start = None
		self.phase = 'mark_start'
		_say('  >> Mark End   %.2fs   (start %.2fs, %.2fs, peak %.3f)  %s' % (t, start, t - start, peak, text))
		try:
			self._write_marked(len(self.utterances))
		except Exception as e:
			_say('  preview pair failed:', e)
		self.line_i += 1
		if self.line_i >= len(self.lines()):
			_say('  take %d done (%d lines). saving…' % (self.take_i, len(self.utterances)))
			self.next_take()
			return
		_say('  next line is up. Mark Start when you begin it.')
		self.refresh()
		self._announce()
		self.listbox.selection_clear(0, 'end')
		self.listbox.selection_set('end')

	def skip_line(self):
		if self.phase == 'arm' or not self.current_text():
			return
		_say('  -- skip line: %s' % self.current_text())
		self.mark_start = None
		self.phase = 'mark_start'
		self.line_i += 1
		if self.line_i >= len(self.lines()):
			self.next_take()
			return
		self.refresh()
		self._announce()

	def undo(self):
		if self.phase == 'mark_stop' and self.mark_start is not None:
			_say('  << undo Mark Start  %.2fs' % self.mark_start)
			self.mark_start = None
			self.phase = 'mark_start'
			self.refresh()
			return
		if not self.utterances:
			return
		u = self.utterances.pop()
		self._drop_pair(len(self.utterances) + 1)
		self.last_stop = self.utterances[-1]['stop'] if self.utterances else 0.0
		self.line_i = max(0, self.line_i - 1)
		self.mark_start = None
		self.phase = 'mark_start' if self.rec.stream is not None else 'arm'
		_say('  << undo  %.2f→%.2f  %s' % (u['start'], u['stop'], u.get('text') or ''))
		self.refresh()
		self._announce()

	def _save_take(self):
		audio, _ = self.take_audio()
		if audio.size < self.rec.sr * 0.4:
			_say('take %d empty, skip save' % self.take_i)
			return None
		if float(np.max(np.abs(audio))) < 1e-4:
			_say('take %d is all zeros — NOT saving. fix the mic first.' % self.take_i)
			return None
		audio = to_16k(audio, self.rec.sr)
		folder = takes_dir(self.data_dir)
		stem = 'take_%02d' % self.take_i
		wav = os.path.join(folder, stem + '.wav')
		jp = os.path.join(folder, stem + '.json')
		save_audio(wav, audio)
		meta = {'sr': SAMPLE_RATE, 'take': self.take_i, 'path': os.path.basename(wav), 'wav': wav, 'prompts': self.lines(), 'utterances': self.utterances}
		write_json(jp, meta)
		_say('saved', wav, 'utterances', len(self.utterances))
		return jp

	def _cache_bg(self, jp):
		def work():
			try:
				while self.rmvpe is None and self.rmvpe_error is None:
					time.sleep(0.05)
				if self.rmvpe is None:
					raise RuntimeError(self.rmvpe_error or 'RMVPE missing')
				cache_take(jp, self.rmvpe)
				_say('cached', os.path.basename(jp))
			except Exception as e:
				_say('cache failed:', e)
			with self.lock:
				self.pending -= 1

		with self.lock:
			self.pending += 1
		threading.Thread(target=work, daemon=True).start()

	def next_take(self):
		jp = self._save_take()
		if jp:
			self._cache_bg(jp)
		self.takes_left -= 1
		if self.takes_left <= 0:
			_say('all takes done. %d clips in the bank.' % N_CLIPS)
			self._wait_then_quit()
			return
		self.take_i += 1
		self.line_i = 0
		self.origin = self.rec.n_samples()
		self.last_stop = 0.0
		self.phase = 'mark_start' if self.rec.stream is not None else 'arm'
		self.mark_start = None
		self.utterances = []
		self.refresh()
		self._announce()

	def finish(self):
		if self.utterances or (self.rec.n_samples() - self.origin) > self.rec.sr:
			jp = self._save_take()
			if jp:
				self._cache_bg(jp)
		self._wait_then_quit()

	def _wait_then_quit(self):
		def wait():
			while True:
				with self.lock:
					left = self.pending
				if left <= 0:
					break
				time.sleep(0.05)
			self.rec.stop()
			self.root.after(0, self.root.destroy)

		threading.Thread(target=wait, daemon=True).start()

	def _tick(self):
		t = self.now_sec()
		self.clock.config(text='%02d:%04.1f' % (int(t) // 60, t % 60))
		self.level_lbl.config(text='peak %.4f   rms %.4f   %s' % (self.rec.peak, self.rec.level, 'SILENT' if self.rec.peak < 1e-4 else 'ok'))
		self.level_lbl.config(fg='#ff8b8b' if self.rec.peak < 1e-4 else '#7cffb2')
		w = min(self.meter_w, int(self.meter_w * min(1.0, self.rec.level * 14)))
		self.meter.coords(self.bar, 0, 0, w, 18)
		self.meter.itemconfig(self.bar, fill='#ff6b9d' if self.phase == 'mark_stop' or self.rec.level > 0.12 else '#7cffb2')
		self.root.after(40, self._tick)

	def _start_mic(self):
		kick_mic_permission()
		try:
			self.rec.start()
		except Exception as e:
			self._set_status('mic failed: %s' % e, '#ff8b8b')
			_say('mic failed:', e)
			return
		self.root.after(1500, self._check_mic)

	def _check_mic(self):
		if self.rec.peak >= 1e-4:
			_say('mic is live  peak=%.4f  (%s @ %d Hz)' % (self.rec.peak, self.rec.name, self.rec.sr))
			self.mic_dead = False
			return
		self.mic_dead = True
		_say('')
		_say('!' * 64)
		_say('MIC IS SILENT.')
		_say('Enable Microphone for Cursor AND Terminal.app:')
		_say('  System Settings → Privacy & Security → Microphone')
		_say('Then quit this, run again, and watch peak rise when you talk.')
		_say('If you use NoMachine / VB-Cable, pass e.g.  --mic 3')
		_say('!' * 64)
		self._set_status('MIC SILENT. Enable Microphone for Cursor/Terminal, then re-run.', '#ff8b8b')
		open_mic_settings()

	def run(self):
		self.refresh()
		_say('')
		_say('LiquidVAD recorder. %d clips, one list. Vary level and distance yourself.' % N_CLIPS)
		_say('A pink window should jump to the front. Watch the peak number — it must move when you talk.')
		_say('Start Recording Data once. Then Mark Start, Mark End, next line. Space does the same. Backspace undoes.')
		_say('Every marked clip is written clean and with pitch + data/noise.wav under data/shown/.')
		self._announce()
		self.root.after(80, self._front)
		self.root.mainloop()
		self.rec.stop()


def main():
	p = argparse.ArgumentParser(description='Record 100 LiquidVAD clips: one mic start, then Mark Start and Mark End on each line')
	p.add_argument('--data', default=DEFAULT_DATA)
	p.add_argument('--takes', type=int, default=len(TAKES), help='continuous takes to record (5 clips each, 20 = 100 clips)')
	p.add_argument('--start-index', type=int, default=None, help='first take number (default: next free, or 1 if old takes are silent)')
	p.add_argument('--mic', default=None, help='input device index or name substring')
	p.add_argument('--rmvpe', default=None)
	p.add_argument('--device', default='mps')
	p.add_argument('--show', action='store_true', help='rewrite data/shown clean+noise pairs from saved takes and exit')
	args = p.parse_args()
	if args.show:
		show_saved(args.data)
		return
	list_mics()
	mic = resolve_mic(args.mic)
	_say('using mic index', mic)
	start = args.start_index
	folder = takes_dir(args.data)
	if start is None:
		existing = [int(n[5:7]) for n in os.listdir(folder) if n.startswith('take_') and n.endswith('.json')] if os.path.isdir(folder) else []
		wavs = [os.path.join(folder, 'take_%02d.wav' % i) for i in existing]
		if existing and wavs and all(wav_is_silent(w) for w in wavs):
			_say('old takes are all silent zeros — overwriting from take 1')
			start = 1
		else:
			start = (max(existing) + 1) if existing else 1
	app = App(args.data, args.takes, args.rmvpe, get_device(args.device), mic=mic)
	app.take_i = start
	app.run()


if __name__ == '__main__':
	main()
