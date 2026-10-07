import numpy as np
import soxr
import soundfile as sf

SAMPLE_RATE = 16000


def to_mono(samples):
	if samples.ndim == 1:
		return samples
	return samples.mean(axis=-1)


def resample_audio(audio, orig_sr, target_sr):
	if orig_sr == target_sr:
		return audio
	return soxr.resample(audio, int(orig_sr), int(target_sr))


def load_audio_np(file, sample_rate=None, mono=True, dtype="float32"):
	"""Path → float ndarray. sample_rate=None keeps native rate; else soxr to target."""
	samples, sr = sf.read(str(file), always_2d=False, dtype="float32")
	samples = np.asarray(samples, dtype=np.float32)
	if mono:
		samples = to_mono(samples)
	if sample_rate is not None and int(sr) != int(sample_rate):
		samples = resample_audio(samples, sr, sample_rate)
		sr = int(sample_rate)
	if dtype != "float32":
		samples = samples.astype(dtype)
	return samples if sample_rate is not None else (samples, int(sr))
