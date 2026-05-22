import json
import os
import uuid
import requests
import subprocess
import torch
from aiflow.utils import get_config
from pydub import AudioSegment
import soundfile as sf
import io
import base64


def load_conditional_imports(config):
	"""Dynamically import modules based on configuration settings."""
	if config["server"]["yuna_text_mode"] == "yuna_vlm":
		from aiflow.models.yuna_vlm.utils import load as load_yuna_text_model
		from aiflow.models.yuna_vlm.generate import stream_generate as stream_generate
		globals()["load_yuna_text_model"] = load_yuna_text_model
		globals()["stream_generate"] = stream_generate

	if config["server"]["yuna_audio_mode"] == "yuna_audio":
		from aiflow.models.yuna_audio.utils import load as load_yuna_audio_model
		from aiflow.models.yuna_audio.utils import audio_read, resample_audio
		from mlx import core as mx
		globals()["load_yuna_audio_model"] = load_yuna_audio_model
		globals()["mx"] = mx
		globals()["audio_read"] = audio_read
		globals()["resample_audio"] = resample_audio

	if config["server"]["yuna_speech_mode"] == "hanasu":
		from aiflow.models.hanasu.models import inference as inference_hanasu
		from aiflow.models.hanasu.models import load_model as load_model_hanasu

		globals()["inference_hanasu"] = inference_hanasu
		globals()["load_model_hanasu"] = load_model_hanasu


class AGIWorker:
	def __init__(self, config=None):
		self.config = get_config() if config is None else config
		self.text_model = None
		self.tokenizer = None
		self.voice_model = None
		self.audio_model = None
		load_conditional_imports(self.config)

	def get_history_text(self, chat_history, text, useHistory, yunaConfig, image_paths, mode="chat"):
		all_image_paths = []

		if useHistory is False:
			return text or "", all_image_paths

		history_str = ""
		if chat_history:
			for m in chat_history:
				role = "yuki" if m["name"].lower() == "yuki" else "yuna"
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
								if "<data>" not in str(message_content):
									message_content = (f"{message_content}<data>{attachment['content']}</data>")
							elif attachment.get("type") == "image":
								img_val = attachment.get("path") or attachment.get("content")
								if img_val and (img_val.startswith("data:image/") or os.path.exists(img_val.lstrip("/"))):
									all_image_paths.append(img_val if img_val.startswith("data:image/") else img_val.lstrip("/"))
									image_count += 1

				if role == "yuki":
					history_str += f"<{role}>{message_content}{'<|vision_start|><|image_pad|><|vision_end|>' * image_count}</{role}>\n"
				else:
					history_str += f"<{role}>{message_content}</{role}>\n"

		if mode == "extend":
			# Fix extending: cleanly strip the closing tag from Yuna's last message
			if history_str.strip().endswith("</yuna>"):
				history_str = history_str.strip()[:-7]
			return history_str, all_image_paths

		elif mode == "second_yuna":
			# Fix consecutive message: history is enclosed, just force new Yuna opening tag
			return f"{history_str.strip()}\n<yuna>", all_image_paths

		# Prevent duplicate <yuki> tags if frontend already pushed the prompt into chat_history
		current_prompt = text or ""
		already_appended = False

		if chat_history and len(chat_history) > 0:
			last_msg = chat_history[-1]
			l_role = "yuki" if last_msg.get("name", "").lower() == "yuki" else "yuna"
			if l_role == "yuki" and last_msg.get("text", "") == current_prompt:
				already_appended = True

		if already_appended:
			final = f"{history_str}<yuna>"
		else:
			current_image_count = len(image_paths or [])
			all_image_paths.extend(image_paths or [])
			final = f"{history_str}<yuki>{current_prompt}{'<|vision_start|><|image_pad|><|vision_end|>' * current_image_count}</yuki>\n<yuna>"

		return final, all_image_paths

	def generate_text(self, text=None, aibo=None, chat_history=None, useHistory=True, yunaConfig=None, image_paths=None, append_current_user=True, continue_from=None, mode="chat", attachments=None):
		if yunaConfig is None:
			yunaConfig = self.config
		self.config = yunaConfig

		# --- PROCESS ATTACHMENTS (TEXT) ---
		if attachments:
			data_blocks = ""
			for att in attachments:
				if att.get("type") == "text" and att.get("content"):
					data_blocks += f"<data>{att['content']}</data>"
			if data_blocks:
				text = f"{text}{data_blocks}" if text else data_blocks

		# --- PARSE DB SYS FILES ---
		sys_tags = ""
		for tag in ["memory", "shujinko", "aibo"]:
			filepath = f"db/{tag}.txt"
			if os.path.exists(filepath):
				with open(filepath, "r", encoding="utf-8") as f:
					content = f.read().strip()
					if content:
						# No interior newlines per instruction
						sys_tags += f"<{tag}>{content}</{tag}>\n"

		# --- BOS TOKEN HANDLING ---
		bos_token = yunaConfig["yuna"]["bos"][0] if yunaConfig["yuna"]["bos"][1] else ""

		# --- MODE HANDLING ---
		all_image_paths = []
		final_prompt = ""
		stop_tokens = yunaConfig["yuna"]["stop"]
		cache_file = None

		if mode == "naked":
			cache_file = "db/cache_naked.safetensors"
			final_prompt = f"{bos_token}\n{text}"
			if image_paths:
				all_image_paths = image_paths
			stop_tokens = []

		elif mode == "loli":
			cache_file = "db/cache_loli.safetensors"
			final_prompt = f"{bos_token}\n{text}"
			if image_paths:
				all_image_paths = image_paths

		elif mode in ["chat", "extend", "second_yuna"]:
			cache_file = "db/cache_chat.safetensors"
			final_prompt, all_image_paths = self.get_history_text(chat_history, text, useHistory, yunaConfig, image_paths, mode=mode)

			if continue_from:
				final_prompt += continue_from

			final_prompt = f"{bos_token}\n{sys_tags}<dialog>\n{final_prompt}"

		# --- EXECUTION ---
		mode_backend = self.config["server"]["yuna_text_mode"]
		print(f"Generating Mode: {mode}, Prompt Length: {len(final_prompt)}")

		kwargs_all = {"max_tokens": yunaConfig["yuna"]["max_new_tokens"], "temperature": yunaConfig["yuna"]["temperature"], "top_p": yunaConfig["yuna"]["top_p"], "top_k": yunaConfig["yuna"]["top_k"], "repetition_penalty": yunaConfig["yuna"]["repetition_penalty"], "repetition_context_size": 4096, "stop_strings": stop_tokens, }

		if mode_backend == "yuna_vlm":
			print(final_prompt)
			print(cache_file)
			response_generator = stream_generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image=all_image_paths, cache_file=cache_file, **kwargs_all)

			def stream_wrapper():
				for chunk in response_generator:
					yield chunk.text

			return stream_wrapper()

		elif mode_backend == "koboldcpp":
			payload = {"temperature": yunaConfig["yuna"]["temperature"], "top_p": yunaConfig["yuna"]["top_p"], "top_k": yunaConfig["yuna"]["top_k"], "min_p": 0.2, "logit_bias": {}, "presence_penalty": 0, "n": 1, "max_context_length": yunaConfig["yuna"]["context_length"], "max_length": yunaConfig["yuna"]["max_new_tokens"], "rep_pen": yunaConfig["yuna"]["repetition_penalty"], "top_a": 0, "typical": 1, "tfs": 0.8, "rep_pen_range": 512, "rep_pen_slope": 0, "sampler_order": [6, 5, 0, 2, 3, 1, 4], "memory": aibo if aibo is not None else "", "trim_stop": True, "genkey": "KCPP9126", "mirostat": 2, "mirostat_tau": 4, "mirostat_eta": 0.3, "dynatemp_range": 0, "dynatemp_exponent": 1, "smoothing_factor": 0, "banned_tokens": [], "render_special": True, "quiet": True, "stop_sequence": stop_tokens, "use_default_badwordsids": False, "bypass_eos": False, "prompt": final_prompt, }
			url = ("http://localhost:5001/api/extra/generate/stream/" if stream else "http://localhost:5001/api/v1/generate/")
			response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, stream=stream)

			if response.status_code == 200:

				def stream_generator():
					for line in response.iter_lines():
						if line:
							decoded_line = line.decode("utf-8")
							if decoded_line.startswith("data: "):
								data = json.loads(decoded_line[6:])
								yield data["token"]

				return stream_generator()

			else:
				return ""
		else:
			return ""

	def load_audio_model(self):
		if self.config["server"]["yuna_audio_mode"] == "yuna_audio":
			self.audio_model = load_yuna_audio_model(self.config["server"]["yuna_audio_model"])

	def load_voice_model(self):
		if self.config["server"]["yuna_speech_mode"] == "hanasu":
			self.voice_model = load_model_hanasu(config_path=self.config["server"]["yuna_speech_model"][0], model_path=self.config["server"]["yuna_speech_model"][1])
			with torch.inference_mode():
				self.voice_model.dec.remove_weight_norm()
			self.voice_model.eval()

	def load_text_model(self):
		if self.config["server"]["yuna_text_mode"] == "yuna_vlm":
			self.text_model, self.tokenizer = load_yuna_text_model(self.config["server"]["yuna_text_model"])

	def export_audio(self, input_file, output_filename):
		AudioSegment.from_file(input_file).export(output_filename, format="mp3")

	def transcribe_audio(self, audio_data):
		# If the input is in-memory bytes from Flask
		if isinstance(audio_data, bytes):
			# Decode bytes via FFmpeg pipe:0 (no disk write)
			samples, orig_sr = audio_read(audio_data)

			# Qwen3/Whisper ASR models require 16kHz
			target_sr = getattr(self.audio_model, 'sample_rate', 16000)
			if orig_sr != target_sr:
				samples = resample_audio(samples, orig_sr, target_sr)

			# Convert to an MLX array. This smartly bypasses the filepath validation checks inside the model's generate() function.
			audio_data = mx.array(samples, dtype=mx.float32)

		return self.audio_model.generate(audio_data).text.strip()

	def speak_text(self, text, output_filename=None):
		output_filename = f"db/{uuid.uuid4()}.m4a"
		mode = self.config["server"]["yuna_speech_mode"]

		if mode == "siri":
			subprocess.run(["say", "-o", "static/audio/temp.aiff", text], check=True)
			subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-threads", "0", "-i", "static/audio/temp.aiff", "-acodec", "alac", output_filename], check=True)
			os.remove("static/audio/temp.aiff")
			return output_filename, None

		elif mode == "siri-pv":
			voice_model = self.config["server"]["yuna_speech_model"][0]
			subprocess.run(["say", "-v", voice_model, "-o", "static/audio/temp.aiff", text], check=True)
			subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-threads", "0", "-i", "static/audio/temp.aiff", "-acodec", "alac", output_filename], check=True)
			os.remove("static/audio/temp.aiff")
			return output_filename, None

		elif mode == "hanasu":
			if not hasattr(self, "voice_model") or self.voice_model is None:
				self.load_voice_model()
			result = inference_hanasu(model=self.voice_model, text=text, device="mps", stream=False)
			wav_io = io.BytesIO()
			sf.write(wav_io, result, 48000, format='WAV')
			wav_b64 = base64.b64encode(wav_io.getvalue()).decode('utf-8')
			data_uri = f"data:audio/wav;base64,{wav_b64}"
			return data_uri, None

	def start(self):
		self.load_text_model()
		self.load_audio_model()
		self.load_voice_model()
