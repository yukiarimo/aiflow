import json
import os
import uuid
import mlx.core as mx
import requests
import torch
import subprocess
from aiflow.utils import get_config, clearText
from pydub import AudioSegment
import soundfile as sf

def load_conditional_imports(config):
    """
    Dynamically import modules based on configuration settings.
    """

    if config["ai"].get("kokoro"): print("Kokoro is not available in this environment.")

    text_mode = config["server"].get("yuna_text_mode")
    if text_mode == "mlx":
        from mlx_lm import generate, load
        globals()['generate'] = generate
        globals()['load'] = load

    if text_mode == "mlxvlm":
        from mlx_vlm import load, generate, stream_generate
        globals()['generate'] = generate
        globals()['load'] = load
        globals()['stream_generate'] = stream_generate

    if config['yuna'].get('audio'):
        if torch.backends.mps.is_available():
            from parakeet_mlx import from_pretrained
            globals()['yunaListenPipe'] = from_pretrained
        elif torch.cuda.is_available(): raise EnvironmentError("Not implemented yet.")

    if config['yuna'].get('hanasu') and config['server'].get('yuna_audio_mode') == 'hanasu':
        from aiflow.models.hanasu.models import inference as inference_hanasu
        from aiflow.models.hanasu.models import load_model as load_model_hanasu
        globals()['inference_hanasu'] = inference_hanasu
        globals()['load_model_hanasu'] = load_model_hanasu

class AGIWorker:
    def __init__(self, config=None):
        self.config = get_config() if config is None else config
        self.text_model = None
        self.tokenizer = None
        self.image_model = None
        self.voice_model = None
        self.audio_model = None
        self.kokoro_model = None
        load_conditional_imports(self.config)

    def get_history_text(self, chat_history, text, useHistory, yunaConfig, image_paths, append_current_user=True, continue_from=None):
        user, asst = yunaConfig["ai"]["names"][0].lower(), yunaConfig["ai"]["names"][1].lower()
        all_image_paths = []  # This list will collect image paths from both history and the current message

        if useHistory is False: final = text
        history_str = ""

        if useHistory and chat_history:
            for m in chat_history:
                role = user if m['name'].lower() == user.lower() else asst
                message_content = m.get('text', '')
                image_count = 0

                # Process attachments within the historical message
                if m.get('data') and isinstance(m.get('data'), list):
                    for attachment in m['data']:
                        if attachment.get('type') == 'text' and attachment.get('content'):
                            message_content = f"{message_content}<data>{attachment['content']}</data>"
                        elif attachment.get('type') == 'image' and attachment.get('path'):
                            image_path = attachment['path'].lstrip('/')
                            if os.path.exists(image_path):
                                all_image_paths.append(image_path)
                                image_count += 1

                if role == user:
                    # Append img tokens for historical messages
                    history_str += f"<{role}>{message_content}{'<|vision_start|><|image_pad|><|vision_end|>' * image_count}</{role}>\n"
                else:
                    history_str += f"<{role}>{message_content}</{role}>\n"

        if append_current_user and useHistory:
            current_prompt = text or ""
            current_image_count = len(image_paths or [])
            all_image_paths.extend(image_paths or [])
            final = f"{history_str}<{user}>{current_prompt}{'<|vision_start|><|image_pad|><|vision_end|>' * current_image_count}</{user}>\n<{asst}>"
        elif useHistory:
            # If not appending current user (e.g. continuing generation),
            # we simply use the history string up to this point.
            final = history_str
        else:
             final = text or ""

        if continue_from:
            final += continue_from

        return final, all_image_paths

    def generate_text(self, text=None, kanojo=None, chat_history=None, useHistory=True, yunaConfig=None, stream=False, image_paths=None, append_current_user=True, continue_from=None, mode='chat', attachments=None):
        if yunaConfig is None: yunaConfig = self.config
        self.config = yunaConfig

        # --- PRE-PROCESSING: Inject Text Attachments ---
        if attachments:
            data_blocks = ""
            for att in attachments:
                if att.get('type') == 'text' and att.get('content'):
                    # Mimicking the <data> structure from your logic
                    data_blocks += f"<data>{att['content']}</data>\n"
            if data_blocks:
                text = f"{data_blocks}{text}" if text else data_blocks

        # --- MODE HANDLING ---
        # "Naked" and "Extend" modes bypass the chat template (get_history_text logic)
        all_image_paths = []
        if mode == 'naked' or mode == 'extend':
            final_prompt = text
            # For MLX VLM, we still need to pass image paths if present in the current request
            if image_paths:
                all_image_paths = image_paths
                # In VLM naked mode, we might need to manually append tokens, but assuming raw text for now
        else:
            final_prompt, all_image_paths = self.get_history_text(chat_history, text, useHistory, yunaConfig, image_paths, append_current_user, continue_from)

            # Add BOS/Dialog wrapper only for Chat Mode
            if kanojo:
                final_prompt = f"{yunaConfig['yuna']['bos'][0]}\n{kanojo}\n<dialog>\n{final_prompt}" if yunaConfig["ai"]["bos"][1] else f"{kanojo}\n<dialog>\n{final_prompt}"
            else:
                final_prompt = f"{yunaConfig['yuna']['bos'][0]}\n<dialog>\n{final_prompt}" if yunaConfig["ai"]["bos"][1] else f"<dialog>\n{final_prompt}"

        # --- EXECUTION ---
        mode_backend = self.config["server"]["yuna_text_mode"]
        stop_tokens = yunaConfig["ai"].get("stop", [])

        kwargs_all = {
            "max_tokens": yunaConfig["ai"]["max_new_tokens"],
            "temperature": yunaConfig["ai"]["temperature"],
            "top_p": yunaConfig["ai"]["top_p"],
            "top_k": yunaConfig["ai"]["top_k"],
            "repetition_penalty": yunaConfig["ai"]["repetition_penalty"],
            "repetition_context_size": 4096,
            "stop_strings": stop_tokens
        }

        if mode_backend == "yunamlx":
            kwargs_yunamlx = {
                **kwargs_all,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "logit_bias": {},
                "eos_token_ids": [self.tokenizer.tokenizer.encode("<|endoftext|>").ids[0]] + [self.tokenizer.tokenizer.encode(token).ids[0] for token in stop_tokens if token],
                "cache": None,
                "stop_strings": stop_tokens
            }

            if stream:
                response_generator = stream_generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image_paths=all_image_paths if all_image_paths else None, video_paths=None, audio_paths=None, stop_strings=stop_tokens if stop_tokens else ["</>"], **kwargs_yunamlx)
                def stream_wrapper():
                    for chunk_text in response_generator: yield chunk_text
                return stream_wrapper()
            else:
                response, _ = generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image_paths=all_image_paths if all_image_paths else None, video_paths=None, audio_paths=None, stop_strings=stop_tokens if stop_tokens else ["</>"], **kwargs_yunamlx)
                return clearText(response)

        if mode_backend == "mlx":
            response_generator = generate(model=self.text_model, tokenizer=self.tokenizer, prompt=final_prompt, verbose=False, eos_tokens=stop_tokens, skip_special_tokens=True, **kwargs_all)
            if stream:
                def stream_wrapper():
                    for chunk_text in response_generator: yield chunk_text
                return stream_wrapper()
            else:
                full_response = "".join(response_generator)
                return clearText(full_response)

        if mode_backend == "mlxvlm":
            # MLX VLM expects a single image for `image` arg usually, or handling list via processor
            # We pass the list directly as `image` based on typical mlx-vlm usage or the list logic in `stream_generate`
            if stream:
                response_generator = stream_generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image=all_image_paths, **kwargs_all)
                def stream_wrapper():
                    for chunk in response_generator:
                        yield chunk.text  # Extract .text here
                return stream_wrapper()
            else:
                response_generator = stream_generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image=all_image_paths, **kwargs_all)
                full_response = "".join([chunk.text for chunk in response_generator])
                return clearText(full_response)

        elif mode_backend == "koboldcpp":
            payload = {
                "temperature": yunaConfig["ai"]["temperature"], "top_p": yunaConfig["ai"]["top_p"],
                "top_k": yunaConfig["ai"]["top_k"], "min_p": 0.2, "logit_bias": {}, "presence_penalty": 0,
                "n": 1, "max_context_length": yunaConfig["ai"]["context_length"],
                "max_length": yunaConfig["ai"]["max_new_tokens"], "rep_pen": yunaConfig["ai"]["repetition_penalty"],
                "top_a": 0, "typical": 1, "tfs": 0.8, "rep_pen_range": 512, "rep_pen_slope": 0,
                "sampler_order": [6, 5, 0, 2, 3, 1, 4], "memory": kanojo if kanojo is not None else "",
                "trim_stop": True, "genkey": "KCPP9126", "mirostat": 2, "mirostat_tau": 4,
                "mirostat_eta": 0.3, "dynatemp_range": 0, "dynatemp_exponent": 1,
                "smoothing_factor": 0, "banned_tokens": [], "render_special": True, "quiet": True,
                "stop_sequence": stop_tokens, "use_default_badwordsids": False, "bypass_eos": False,
                "prompt": final_prompt
            }
            url = "http://localhost:5001/api/extra/generate/stream/" if stream else "http://localhost:5001/api/v1/generate/"
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, stream=stream)

            if response.status_code == 200:
                if stream:
                    def stream_generator():
                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith('data: '):
                                    data = json.loads(decoded_line[6:])
                                    yield data['token']
                    return stream_generator()
                else:
                    response_json = response.json()
                    if "results" in response_json and response_json["results"]:
                        resp = response_json["results"][0].get('text', '')
                        return clearText(resp)
                    else: return ''
            else: return ''
        else: return ''

    def load_audio_model(self): self.yunaListen = yunaListenPipe("mlx-community/parakeet-tdt-0.6b-v3", dtype=mx.float16)

    def load_voice_model(self):
        if self.config["server"]["yuna_audio_mode"] == "hanasu":
            self.voice_model = load_model_hanasu(config_path=self.config['server']['voice_default_model'][0], model_path=self.config['server']['voice_default_model'][1])

    def load_text_model(self):
        mode = self.config["server"].get("yuna_text_mode")
        if mode == "llamacpp":
            # Assuming Llama import is available in environment
            from llama_cpp import Llama
            self.text_model = Llama(
                model_path=self.config['server']['yuna_default_model'][0],
                n_ctx=self.config["ai"]["context_length"],
                last_n_tokens_size=self.config["ai"]["last_n_tokens_size"],
                seed=self.config["ai"]["seed"],
                n_batch=self.config["ai"]["batch_size"],
                n_gpu_layers=self.config["ai"]["gpu_layers"],
                n_threads=self.config["ai"]["threads"],
                use_mmap=self.config["ai"]["use_mmap"],
                use_mlock=self.config["ai"]["use_mlock"],
                flash_attn=self.config["ai"]["flash_attn"],
                offload_kqv=self.config["ai"]["offload_kqv"],
                verbose=False
            )
        elif mode == "mlx": self.text_model, self.tokenizer = load(self.config['server']['yuna_default_model'][0])
        elif mode == "mlxvlm": self.text_model, self.tokenizer = load(self.config['server']['yuna_default_model'][0])

    def load_kokoro_model(self, config, model_path): print("Kokoro is not available in this environment.")
    def export_audio(self, input_file, output_filename): AudioSegment.from_file(input_file).export(output_filename, format="mp3")
    def transcribe_audio(self, audio_file): return self.yunaListen.transcribe(audio_file).text.strip()

    def speak_text(self, text, output_filename=None):
        output_filename = f"static/audio/{uuid.uuid4()}.wav"
        mode = self.config['server']['yuna_audio_mode']

        if mode == 'siri':
            os.system(f'say -o "static/audio/temp.aiff" {repr(text)}')
            subprocess.run(['ffmpeg', '-y', '-i', 'static/audio/temp.aiff', output_filename], check=True, capture_output=True)
            os.remove("static/audio/temp.aiff")

            with open(output_filename, 'rb') as f:
                audio_bytes = f.read()
            return output_filename, audio_bytes

        elif mode == "siri-pv":
            voice_model = self.config['server']['voice_default_model'][0]
            os.system(f'say -v {voice_model} -o "static/audio/temp.aiff" {repr(text)}')
            subprocess.run(['ffmpeg', '-y', '-i', 'static/audio/temp.aiff', output_filename], check=True, capture_output=True)
            os.remove("static/audio/temp.aiff")

            with open(output_filename, 'rb') as f:
                audio_bytes = f.read()
            return output_filename, audio_bytes

        elif mode == "hanasu":
            result = inference_hanasu(
                model=self.voice_model,
                text=text,
                sid=0,
                device=self.config['server']['device'],
                stream=False,
            )

            # Save to file
            sf.write(output_filename, result, 48000)

            # Get bytes
            import io
            byte_io = io.BytesIO()
            sf.write(byte_io, result, 48000, format='WAV')
            audio_bytes = byte_io.getvalue()

            return output_filename, audio_bytes

        return None, None

    def processTextFile(self, text_file, question):
        from aiflow.utils import calculate_similarity
        from aiflow.models.kokoro.model import EmbeddingGenerator
        import re

        with open(text_file, 'r', encoding='utf-8') as f: text = f.read()

        def split_sentences(text):
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            sentences = re.split(r'([.!?…])', text)
            result = []
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i].strip()
                if sentence:
                    punctuation = sentences[i + 1] if i + 1 < len(sentences) else ''
                    result.append(sentence + punctuation)
            if len(sentences) % 2 == 1 and sentences[-1].strip(): result.append(sentences[-1].strip())
            return result

        sentences = split_sentences(text)
        print(f"Split into {len(sentences)} sentences")

        config = {"embedding_model_path": self.config['server']['yuna_default_model'][0], "embedding_dimensions": 4096}
        embed_generator = EmbeddingGenerator(config)
        question_embedding = embed_generator.get_embedding(question)

        similarities = []
        for sentence in sentences:
            sentence_embedding = embed_generator.get_embedding(sentence)
            similarity = calculate_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_50_percent = similarities[:len(similarities)//2]
        filtered_sentences = [item[0] for item in top_50_percent]

        print(f"Filtered to {len(filtered_sentences)} most relevant sentences")
        return ' '.join(filtered_sentences)

    def start(self):
        # We check config settings to load models at startup
        if self.config["settings"].get("mind", True): self.load_text_model()
        if self.config["ai"].get("audio"): self.load_audio_model()
        if self.config["ai"].get("hanasu"): self.load_voice_model()
        if self.config["ai"].get("kokoro"): self.load_kokoro_model()