import json
import os
import uuid
import requests
import subprocess
import torch
from aiflow.utils import get_config
from pydub import AudioSegment
import soundfile as sf


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

	def get_history_text(self, chat_history, text, useHistory, yunaConfig, image_paths, append_current_user=True, continue_from=None):
		all_image_paths = []

		if useHistory is False:
			final = text
		history_str = ""

		if useHistory and chat_history:
			for m in chat_history:
				role = "yuki" if m["name"].lower() == "yuki" else "yuna"
				message_content = m.get("text", "")
				image_count = 0

				if m.get("images") and isinstance(m.get("images"), list):
					for attachment in m["images"]:
						if isinstance(attachment, str):
							if os.path.exists(attachment.lstrip("/")):
								all_image_paths.append(attachment.lstrip("/"))
								image_count += 1
						elif isinstance(attachment, dict):
							if attachment.get("type") == "text" and attachment.get("content"):
								if "<data>" not in str(message_content):
									message_content = (f"{message_content}<data>{attachment['content']}</data>")
							elif attachment.get("type") == "image" and attachment.get("path"):
								image_path = attachment["path"].lstrip("/")
								if os.path.exists(image_path):
									all_image_paths.append(image_path)
									image_count += 1

				if role == "yuki":
					history_str += f"<{role}>{message_content}{'<|vision_start|><|image_pad|><|vision_end|>' * image_count}</{role}>\n"
				else:
					history_str += f"<{role}>{message_content}</{role}>\n"

		if append_current_user and useHistory:
			current_prompt = text or ""
			current_image_count = len(image_paths or [])
			all_image_paths.extend(image_paths or [])
			final = f"{history_str}<yuki>{current_prompt}{'<|vision_start|><|image_pad|><|vision_end|>' * current_image_count}</yuki>\n<yuna>"
		elif useHistory:
			final = f"{history_str}<yuna>"
		else:
			final = text or ""

		if continue_from:
			final += continue_from

		return final, all_image_paths

	def generate_text(self, text=None, aibo=None, chat_history=None, useHistory=True, yunaConfig=None, image_paths=None, append_current_user=True, continue_from=None, mode="chat", attachments=None):
		if yunaConfig is None:
			yunaConfig = self.config
		self.config = yunaConfig

		# --- PROCESS ATTACHMENTS (TEXT) ---
		# Attachments are appended to the text to appear inside the <yuki> block later
		if attachments:
			data_blocks = ""
			for att in attachments:
				if att.get("type") == "text" and att.get("content"):
					data_blocks += f"<data>{att['content']}</data>"
			if data_blocks:
				text = f"{text}{data_blocks}" if text else data_blocks

		# --- BOS TOKEN HANDLING ---
		bos_token = yunaConfig["yuna"]["bos"][0] if yunaConfig["yuna"]["bos"][1] else ""

		# --- MODE HANDLING ---
		all_image_paths = []
		final_prompt = ""
		stop_tokens = yunaConfig["yuna"]["stop"]

		# Define cache file based on mode
		cache_file = None

		if mode == "naked":
			cache_file = "db/cache_naked.safetensors"

			# Naked: <bos>{text} - NO STOP TOKENS used (except EOS if model defaults)
			final_prompt = f"{bos_token}{text}"
			if image_paths:
				all_image_paths = image_paths
			# User requested NO stop tokens for naked mode
			stop_tokens = []

		elif mode == "loli":
			cache_file = "db/cache_loli.safetensors"

			# LoliConnect: <bos>{pre_prompt_processed}
			# Note: The 'text' argument here comes from index.py pre-processed (replacing {input} etc)
			# The structure is already baked into 'text' by the server route logic.
			final_prompt = f"{bos_token}{text}"
			if image_paths:
				all_image_paths = image_paths
			# User requested stop tokens FOR loli connect
			stop_tokens = yunaConfig["yuna"]["stop"]

		elif mode == "extend":
			cache_file = "db/cache_chat.safetensors"

			# Reconstruct history but leave the last turn open
			# Assuming 'text' passed in is just raw history text, we need to parse/format it properly
			# Or if chat_history is passed, use get_history_text but strip the closing tag
			if chat_history:
				final_prompt, _ = self.get_history_text(chat_history, "", useHistory, yunaConfig, [], append_current_user=False)
				# Strip the last newline and closing tag (e.g. </yuna>) to allow continuation
				# Simple heuristic: remove last newline and last 7 chars (</yuna>)
				if final_prompt.strip().endswith("</yuna>"):
					final_prompt = final_prompt.strip()[:-7]
			else:
				# Fallback to raw text if no structured history provided, assuming user knows what they are doing
				final_prompt = text

		else:  # Chat Mode
			cache_file = "db/cache_chat.safetensors"

			final_prompt, all_image_paths = self.get_history_text(chat_history, text, useHistory, yunaConfig, image_paths, append_current_user, continue_from)
			# Standard Chat Template
			if aibo:
				final_prompt = f"{bos_token}\n{aibo}\n<dialog>\n{final_prompt}"
			else:
				final_prompt = f"{bos_token}\n<dialog>\n{final_prompt}"
			stop_tokens = yunaConfig["yuna"]["stop"]

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

	def transcribe_audio(self, audio_file):
		return self.audio_model.generate(audio_file).text.strip()

	def speak_text(self, text, output_filename=None):
		output_filename = f"static/audio/{uuid.uuid4()}.wav"
		mode = self.config["server"]["yuna_speech_mode"]

		if mode == "siri":
			os.system(f'say -o "static/audio/temp.aiff" {repr(text)}')
			subprocess.run(["ffmpeg", "-y", "-i", "static/audio/temp.aiff", output_filename], check=True, capture_output=True)
			os.remove("static/audio/temp.aiff")
			return output_filename, None

		elif mode == "siri-pv":
			voice_model = self.config["server"]["yuna_speech_model"][0]
			os.system(f'say -v {voice_model} -o "static/audio/temp.aiff" {repr(text)}')
			subprocess.run(["ffmpeg", "-y", "-i", "static/audio/temp.aiff", output_filename], check=True, capture_output=True)
			os.remove("static/audio/temp.aiff")
			return output_filename, None

		elif mode == "hanasu":
			if not hasattr(self, "voice_model") or self.voice_model is None:
				self.load_voice_model()
			result = inference_hanasu(model=self.voice_model, text=text, device="mps", stream=False)
			sf.write(output_filename, result, 48000)
			return output_filename, None

		return None, None

	def start(self):
		self.load_text_model()
		self.load_audio_model()
		self.load_voice_model()
