import os
import uuid
import subprocess
import torch
from aiflow.utils import get_config
import soundfile as sf
import io
import base64


def _shots_for(sentence, bank, keep=4):
	"""First examples always. A later example is used only when the sentence has its long words."""
	import re

	def words(text):
		return set(re.findall(r"[^\W\d_]{4,}", text.lower(), flags=re.UNICODE))

	base = list(bank[:keep])
	sw = words(sentence)
	for pair in bank[keep:]:
		need = words(pair[0])
		long = {w for w in need if len(w) >= 5}
		if long and long <= sw:
			base.append(pair)
		elif not long and need & sw:
			base.append(pair)
	return base


def load_conditional_imports(config):
	"""Dynamically import modules based on configuration settings."""
	if config["server"]["yuna_text_mode"] == "yuna_vlm":
		from aiflow.models.yuna_vlm.utils import load as load_yuna_text_model
		from aiflow.models.yuna_vlm.generate import stream_generate as stream_generate
		globals()["load_yuna_text_model"] = load_yuna_text_model
		globals()["stream_generate"] = stream_generate

	if config["server"]["yuna_audio_mode"] == "yuna_audio":
		from aiflow.models.yuna_audio.utils import load as load_yuna_audio_model
		from mlx import core as mx
		globals()["load_yuna_audio_model"] = load_yuna_audio_model
		globals()["mx"] = mx

	if config["server"]["yuna_speech_mode"] == "vits":
		from aiflow.models.yuna_speech.vits.models import inference as inference_vits
		from aiflow.models.yuna_speech.vits.models import load_model as load_model_vits

		globals()["inference_vits"] = inference_vits
		globals()["load_model_vits"] = load_model_vits


class AGIWorker:
	def __init__(self, config=None):
		self.config = get_config() if config is None else config
		self.text_model = None
		self.tokenizer = None
		self.voice_model = None
		self.audio_model = None
		load_conditional_imports(self.config)

	def _load_sys_tags(self):
		"""Load memory / shujinko / aibo from lib/yuna.json. Empty strings are omitted."""
		from aiflow.utils import bag_load, yuna_json_path
		store = self.config.get("server", {}).get("yuna_json") or yuna_json_path()
		data = bag_load(store)
		sys_tags = ""
		for tag in ["memory", "shujinko", "aibo"]:
			content = data.get(tag)
			if isinstance(content, str) and content.strip():
				sys_tags += f"<{tag}>{content.strip()}</{tag}>\n"
		return sys_tags

	def get_history_text(self, chat_history, text, image_paths, mode="chat", audio_paths=None):
		all_image_paths = []
		all_audio_paths = list(audio_paths or [])
		history_str = ""
		current_prompt = text or ""
		already_appended = False

		if mode == "chat" and chat_history:
			last_msg = chat_history[-1]
			if last_msg.get("name", "").lower() == "yuki" and last_msg.get("text", "") == current_prompt:
				already_appended = True
				# Turn already in history; client may have persisted without the attachments just sent — merge on a copy so data URIs never leak back, then let the loop emit pads (one <|image_pad|> per path; VLM rejects any other ratio).
				persisted = last_msg.get("images") if isinstance(last_msg.get("images"), list) else []
				seen = {a if isinstance(a, str) else (a.get("path") or a.get("content")) for a in persisted if isinstance(a, (str, dict))}
				merged = persisted + [p for p in (image_paths or []) if p not in seen]
				chat_history = list(chat_history[:-1]) + [dict(last_msg, images=merged)]

		if chat_history:
			for m in chat_history:
				if m.get("summarized"):
					continue
				name_lower = m["name"].lower()
				role = "yuki" if name_lower == "yuki" else "yuna"
				message_content = m.get("text", "")
				image_count = 0

				if m.get("images") and isinstance(m.get("images"), list):
					for attachment in m["images"]:
						if isinstance(attachment, str):
							if attachment.startswith("data:image/") or os.path.exists(attachment.lstrip("/")):
								all_image_paths.append(attachment if attachment.startswith("data:image/") else attachment.lstrip("/"))
								image_count += 1
						elif isinstance(attachment, dict):
							if attachment.get("type") == "text" and attachment.get("content"):
								block = f"<data>{attachment['content']}</data>"
								if block not in str(message_content):
									message_content = f"{message_content}{block}"
							elif attachment.get("type") == "image":
								img_val = attachment.get("path") or attachment.get("content")
								if img_val and (img_val.startswith("data:image/") or os.path.exists(img_val.lstrip("/"))):
									all_image_paths.append(img_val if img_val.startswith("data:image/") else img_val.lstrip("/"))
									image_count += 1
							elif attachment.get("type") == "audio":
								aud_val = attachment.get("path") or attachment.get("content")
								if aud_val:
									all_audio_paths.append(aud_val)

				if role == "yuki":
					vision_pads = "<|vision_start|><|image_pad|><|vision_end|>" * image_count  # pads at end of <yuki>; <data> is text/files only — never wrap modality
					audio_count = 0
					if m.get("images") and isinstance(m.get("images"), list):
						for attachment in m["images"]:
							if isinstance(attachment, dict) and attachment.get("type") == "audio":
								audio_count += 1
					audio_pads = "<|audio_start|><|audio_pad|><|audio_end|>" * audio_count
					history_str += f"<yuki>{message_content}{vision_pads}{audio_pads}</yuki>\n"
				else:
					history_str += f"<yuna>{message_content}</yuna>\n"

		if mode == "extend":
			stripped = history_str.rstrip()
			if stripped.endswith("</yuna>"):
				stripped = stripped[:-len("</yuna>")]
			return stripped, all_image_paths, all_audio_paths

		if mode == "second_yuna":
			return f"{history_str}<yuna>", all_image_paths, all_audio_paths

		if already_appended:
			if all_audio_paths and "<|audio_start|>" not in history_str and history_str.rstrip().endswith("</yuki>"):
				pad = "<|audio_start|><|audio_pad|><|audio_end|>" * len(all_audio_paths)
				history_str = history_str.rstrip()[:-len("</yuki>")] + pad + "</yuki>\n"
			return f"{history_str}<yuna>", all_image_paths, all_audio_paths

		current_image_count = len(image_paths or [])
		all_image_paths.extend(image_paths or [])
		vision_pads = "<|vision_start|><|image_pad|><|vision_end|>" * current_image_count if current_image_count > 0 else ""
		audio_pads = "<|audio_start|><|audio_pad|><|audio_end|>" * len(all_audio_paths) if all_audio_paths else ""
		return f"{history_str}<yuki>{current_prompt}{vision_pads}{audio_pads}</yuki>\n<yuna>", all_image_paths, all_audio_paths

	def generate_text(self, text=None, chat_history=None, yunaConfig=None, image_paths=None, mode="chat", attachments=None, audio_paths=None):
		if yunaConfig is None:
			yunaConfig = self.config
		self.config = yunaConfig

		audio_paths = list(audio_paths or [])
		if attachments:
			files = []
			for att in attachments:
				if att.get("type") == "text" and att.get("content"):
					files.append(f"{att.get('name') or 'file'}\n{att['content']}")
				elif att.get("type") == "audio":
					src = att.get("path") or att.get("content")
					if src:
						audio_paths.append(src)
			if files and "<data>" not in (text or ""):
				block = "<data>" + "\n\n".join(files) + "</data>"
				text = f"{text}{block}" if text else block

		bos_token = yunaConfig["yuna"]["bos"][0] if yunaConfig["yuna"]["bos"][1] else ""
		bos_prefix = f"{bos_token}\n" if bos_token else ""
		all_image_paths = []
		final_prompt = ""
		stop_tokens = yunaConfig["yuna"]["stop"]
		cache_file = None

		if mode == "naked":
			cache_file = "lib/cache_naked.safetensors"
			final_prompt = f"{bos_token}{text or ''}"
			if image_paths:
				all_image_paths = image_paths
			stop_tokens = ["</put>"] if final_prompt.rstrip().endswith("<put>") else []

		elif mode == "sage":
			cache_file = "lib/cache_sage.safetensors"
			final_prompt = f"{bos_prefix}{text or ''}"
			if image_paths:
				all_image_paths = image_paths
			stop_tokens = ["</action>", "</yuna>", "<yuki>"]  # chat stops include <yuna>/<data>; phase 2 starts after <data> and she often opens another <yuna> — those stops cut the reply to nothing

		elif mode in ["chat", "extend", "second_yuna"]:
			cache_file = "lib/cache_chat.safetensors"
			sys_tags = self._load_sys_tags()
			final_prompt, all_image_paths, audio_paths = self.get_history_text(chat_history, text, image_paths, mode=mode, audio_paths=audio_paths)
			final_prompt = f"{bos_prefix}{sys_tags}<dialog>\n{final_prompt}"  # <|endoftext|>\n<memory>…</memory>\n<shujinko>…</shujinko>\n<aibo>…</aibo>\n<dialog>\n{…}

		mode_backend = self.config["server"]["yuna_text_mode"]
		print(f"\n----- Yuna prompt ({mode}) -----\n{final_prompt}\n----- end prompt -----\n", flush=True)
		kwargs_all = {"max_tokens": yunaConfig["yuna"]["max_new_tokens"], "temperature": yunaConfig["yuna"]["temperature"], "top_p": yunaConfig["yuna"]["top_p"], "top_k": yunaConfig["yuna"]["top_k"], "repetition_penalty": yunaConfig["yuna"]["repetition_penalty"], "repetition_context_size": 4096, "stop_strings": stop_tokens, }

		if mode_backend == "yuna_vlm":
			response_generator = stream_generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image=all_image_paths, audio=audio_paths or None, cache_file=cache_file, **kwargs_all)

			def stream_wrapper():
				for chunk in response_generator:
					yield chunk.text

			return stream_wrapper()

		return ""

	def load_audio_model(self):
		if self.config["server"]["yuna_audio_mode"] == "yuna_audio":
			self.audio_model = load_yuna_audio_model(self.config["server"]["yuna_audio_model"])

	def load_voice_model(self):
		if self.config["server"]["yuna_speech_mode"] == "vits":
			backend = self.config["server"]["yuna_speech_model"][2] if len(self.config["server"]["yuna_speech_model"]) > 2 else "pytorch"
			if backend == "torch":
				backend = "pytorch"
			if backend == "pytorch":
				self.voice_model = load_model_vits(config_path=self.config["server"]["yuna_speech_model"][0], model_path=self.config["server"]["yuna_speech_model"][1], backend="pytorch")
				with torch.inference_mode():
					self.voice_model.dec.remove_weight_norm()
				self.voice_model.eval()
			elif backend in {"coreml", "coreai", "onnx"}:
				self.voice_model = load_model_vits(config_path=self.config["server"]["yuna_speech_model"][0], model_path=self.config["server"]["yuna_speech_model"][1], backend=backend)

	def load_text_model(self):
		if self.config["server"]["yuna_text_mode"] == "yuna_vlm":
			self.text_model, self.tokenizer = load_yuna_text_model(self.config["server"]["yuna_text_model"])

	def _prepare_audio_for_model(self, audio_data):
		"""Decode raw upload bytes → mono float @ model SR (ffmpeg one-pass), return mx.array."""
		if isinstance(audio_data, bytes):
			from aiflow.models.yuna_audio.utils import load_audio_np
			target_sr = getattr(self.audio_model, "sample_rate", 16000)
			return mx.array(load_audio_np(audio_data, sample_rate=target_sr, mono=True), dtype=mx.float32)
		return audio_data

	@staticmethod
	def _auto_max_tokens(audio_samples, sample_rate=16000, min_tokens=8192, tokens_per_sec=12):
		"""Scale max_tokens to the audio duration so long lectures aren't truncated. Qwen3-ASR emits ~5–8 BPE tokens/sec of speech; we budget 12/s for safety."""
		try:
			duration_sec = float(audio_samples.shape[0]) / float(sample_rate)
		except Exception:
			return min_tokens
		return max(min_tokens, int(duration_sec * tokens_per_sec))

	def _avl_transcribe_prompt(self, language=None):
		"""Raw AVL asr-text turn. ChatML atoms are blank on the remade AVL; strings still BPE."""
		lang = f"language {language}<asr_text>" if language else "<asr_text>"
		return f"<|audio_start|><|audio_pad|><|audio_end|>\n{lang}"

	def translate_text(self, user_input):
		"""One sentence at a time, using that language's shots from the skill. Paragraph breaks stay."""
		import re
		from aiflow.utils import bag_load, yuna_json_path
		store = self.config.get("server", {}).get("yuna_json") or yuna_json_path()
		shots = (bag_load(store).get("translate_shots") or {})
		raw = (user_input or "").strip()
		lang, _, source = raw.partition(":")
		lang, source = lang.strip(), source.strip()
		bank = shots.get(lang) or []
		if not source or not bank:
			return raw
		rule = "You only translate, then you stop. Write only the translation in the named language. Do not add anything."

		def one(sentence):
			picked = _shots_for(sentence, bank)
			head = "".join(f"<yuki>Please translate this text into: <data>{lang}: {src}</data></yuki>\n<yuna>{dst}</yuna>\n" for src, dst in picked)
			prompt = f"<|endoftext|>\n<aibo>{rule}</aibo>\n<dialog>\n{head}<yuki>Please translate this text into: <data>{lang}: {sentence}</data></yuki>\n<yuna>"
			parts = []
			for piece in stream_generate(model=self.text_model, processor=self.tokenizer, prompt=prompt, max_tokens=32, temperature=0.0, top_p=1.0, repetition_penalty=1.0, repetition_context_size=64, stop_strings=["</yuna>", "<yuki>", "<action>", "<data>", "\n"]):
				parts.append(piece.text if hasattr(piece, "text") else str(piece))
			return "".join(parts).strip().split("\n")[0].strip()

		paras = []
		for para in re.split(r"\n\s*\n", source):
			sents = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s*", para.strip()) if s.strip()]
			paras.append(" ".join(one(s) for s in sents))
		return "\n\n".join(p for p in paras if p)

	def transcribe_audio(self, audio_data, chunk_duration=600.0, language=None):
		"""Blocking transcription. AVL text model wins when it has an audio tower."""
		if self.text_model is not None and hasattr(self.text_model, "audio_tower"):
			audio_data = self._prepare_audio_for_model(audio_data)
			prompt = self._avl_transcribe_prompt(language)
			max_tokens = self._auto_max_tokens(audio_data)
			chunks = []
			for piece in stream_generate(model=self.text_model, processor=self.tokenizer, prompt=prompt, audio=audio_data, max_tokens=max_tokens, temperature=0.1, stop_strings=["</yuna>", "<|im_end|>"], ):
				chunks.append(piece.text)
			text = (chunks[-1] if chunks else "").strip()
			if "<asr_text>" in text:
				text = text.split("<asr_text>", 1)[-1].strip()
			return text
		audio_data = self._prepare_audio_for_model(audio_data)
		max_tokens = self._auto_max_tokens(audio_data)
		return self.audio_model.generate(audio_data, max_tokens=max_tokens, chunk_duration=chunk_duration).text.strip()

	def transcribe_audio_stream(self, audio_data, chunk_duration=120.0):
		"""Voice transcription. Cut on silence, about two minutes each, then concatenate."""
		audio_data = self._prepare_audio_for_model(audio_data)
		pieces = self._silence_pieces(audio_data, max_sec=chunk_duration)

		class _Piece:
			def __init__(self, text, is_final):
				self.text, self.is_final = text, is_final

		def gen():
			for i, piece in enumerate(pieces):
				text = self.transcribe_audio(piece, chunk_duration=chunk_duration).strip()
				yield _Piece(text, i == len(pieces) - 1)

		return gen()

	@staticmethod
	def _silence_pieces(audio, max_sec=120.0, sr=16000):
		import numpy as np
		flat = np.asarray(audio).reshape(-1).astype("float32")
		max_n = max(sr, int(max_sec * sr))
		if flat.size <= max_n:
			return [audio]
		win, need, thresh = max(1, sr // 50), int(0.28 * sr), 0.012
		quiet, run, i = [], 0, 0
		while i + win <= flat.size:
			rms = float(np.sqrt(np.mean(flat[i:i + win]**2)))
			if rms < thresh:
				run += win
				if run >= need:
					quiet.append(i)
			else:
				run = 0
			i += win
		spans, start = [], 0
		while start < flat.size:
			limit = min(flat.size, start + max_n)
			if limit == flat.size:
				spans.append((start, flat.size))
				break
			cut = next((q for q in reversed(quiet) if start + sr < q <= limit), limit)
			end = max(start + 1, cut)
			spans.append((start, end))
			start = end
		import mlx.core as mx
		return [mx.array(flat[a:b]) for a, b in spans]

	def _siri_to_m4a(self, text, output_filename, voice=None):
		"""say → temp AIFF beside the m4a, then ffmpeg ALAC. Creates parents."""
		import tempfile
		out_dir = os.path.dirname(os.path.abspath(output_filename)) or tempfile.gettempdir()
		os.makedirs(out_dir, exist_ok=True)
		temp_aiff = os.path.join(out_dir, f"{uuid.uuid4()}.aiff")
		cmd = ["say", "-v", voice, "-o", temp_aiff, text] if voice else ["say", "-o", temp_aiff, text]
		subprocess.run(cmd, check=True)
		subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-threads", "0", "-i", temp_aiff, "-acodec", "alac", output_filename], check=True)
		os.remove(temp_aiff)
		return output_filename

	def speak_text(self, text, output_filename=None):
		"""Always ``(audio, error)`` — path or data URI + error string. Modes outside siri/siri-pv/vits mean speech off (e.g. ``vits1``); bare ``None`` made unpackers raise ``TypeError``. Default file output is process temp dir."""
		mode = self.config["server"]["yuna_speech_mode"]
		if mode not in ("siri", "siri-pv", "vits"):
			return None, f"speech is off (yuna_speech_mode={mode!r})"

		if mode in ("siri", "siri-pv"):
			import tempfile
			if output_filename is None:
				output_filename = os.path.join(tempfile.gettempdir(), f"yuna-{uuid.uuid4()}.m4a")
			voice = self.config["server"]["yuna_speech_model"][0] if mode == "siri-pv" else None
			return self._siri_to_m4a(text, output_filename, voice=voice), None

		if not hasattr(self, "voice_model") or self.voice_model is None:  # vits — data URI, not a file path
			self.load_voice_model()
		backend = self.config["server"]["yuna_speech_model"][2] if len(self.config["server"]["yuna_speech_model"]) > 2 else "pytorch"
		if backend == "torch":
			backend = "pytorch"
		result = inference_vits(model=self.voice_model, text=text, device="mps", stream=False, backend=backend, config_path=self.config["server"]["yuna_speech_model"][0], style_ref_wav=getattr(self, "style_ref_wav", None))
		wav_io = io.BytesIO()
		sf.write(wav_io, result, 48000, format='WAV')
		wav_b64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
		return f"data:audio/wav;base64,{wav_b64}", None

	def start(self):
		self.load_text_model()
		self.load_audio_model()
		self.load_voice_model()
