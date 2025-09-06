import json
import os
import uuid
import torch
import requests
from aiflow.utils import get_config, clearText, search_web
from pydub import AudioSegment
from scipy.io.wavfile import write
import re

def load_conditional_imports(config):
    """
    Dynamically import modules based on configuration settings.

    Args:
        config (dict): Application configuration dictionary
    """

    if config["ai"].get("kokoro"): print("Kokoro is not available in this environment.")

    text_mode = config["server"].get("yuna_text_mode")
    if text_mode == "mlx":
        from mlx_lm import generate, load
        globals()['generate'] = generate
        globals()['load'] = load

    if text_mode == "mlxvlm":
        from mlx_vlm import load, generate
        globals()['generate'] = generate
        globals()['load'] = load

    if config['ai'].get('audio'):
        from transformers import pipeline
        globals()['pipeline'] = pipeline

    audio_mode = config['server'].get('yuna_audio_mode')
    print(f"Audio mode: {audio_mode}")
    if audio_mode == "hanasu":
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

    def get_history_text(self, chat_history, text, useHistory, yunaConfig):
        user, asst = yunaConfig["ai"]["names"][0].lower(), yunaConfig["ai"]["names"][1].lower()

        # MODIFIED: Rebuild history string with awareness of attachments
        history_str = ""
        if useHistory and chat_history:
            for m in chat_history:
                role = user if m['name'].lower() == user else asst

                # Start with the original text of the message
                message_content = m.get('text', '')

                history_str += f"<{role}>{message_content}</{role}>\n"

        current_prompt = text.get('text') if isinstance(text, dict) else text
        final = f"{history_str}<{user}>{current_prompt}</{user}>\n<{asst}>"
        return final

    def generate_text(self, text=None, kanojo=None, chat_history=None, useHistory=True, yunaConfig=None, stream=False, image_path=None):
        self.config = yunaConfig or self.config
        mode = self.config["server"]["yuna_text_mode"]
        if useHistory: final_prompt = self.get_history_text(chat_history, clearText(text), useHistory, yunaConfig)
        else: final_prompt = clearText(text)

        final_prompt = "<bos>\n<dialog>\n" + final_prompt # remove <bos> ???

        kwargs = {
            "max_tokens": yunaConfig["ai"]["max_new_tokens"],
            "temperature": yunaConfig["ai"]["temperature"],
            "prefill_step_size": 2048,
            "kv_group_size": 16,
            "quantized_kv_start": 0,
            "top_p": yunaConfig["ai"]["top_p"],
            "top_k": yunaConfig["ai"]["top_k"],
            "repetition_penalty": yunaConfig["ai"]["repetition_penalty"],
            "repetition_context_size": 128,
            "eos_tokens": ["</yuna>", "</yuki>", "</start_of_image>"],
            "skip_special_tokens": True,
            "stop": yunaConfig["ai"]["stop"]
        }

        if mode == "mlx":
            text = generate(model=self.text_model, tokenizer=self.tokenizer, prompt=final_prompt, verbose=True, **kwargs)
            return clearText(text.text)
        elif mode == "mlxvlm":
            if image_path is None:
                text = generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, verbose=False, **kwargs)
            else:
                # insert image token "<start_of_image>" before the last only <yuna> tag
                yuna_tag = yunaConfig['ai']['names'][1].lower()
                pattern = f'<{yuna_tag}>'
                matches = list(re.finditer(pattern, final_prompt))
                if matches:
                    last_match = matches[-1]
                    final_prompt = final_prompt[:last_match.start()] + f'<start_of_image><{yuna_tag}>' + final_prompt[last_match.end():]
                print(final_prompt)
                text = generate(model=self.text_model, processor=self.tokenizer, prompt=final_prompt, image=image_path, verbose=False, **kwargs)
            return clearText(text.text)
        elif mode == "koboldcpp":
            common_payload = {
                "temperature": yunaConfig["ai"]["temperature"],
                "top_p": yunaConfig["ai"]["top_p"],
                "top_k": yunaConfig["ai"]["top_k"],
                "min_p": 0.2,
                "logit_bias": {},
                "presence_penalty": 0,
            }
            specific = {
                "n": 1,
                "max_context_length": yunaConfig["ai"]["context_length"],
                "max_length": yunaConfig["ai"]["max_new_tokens"],
                "rep_pen": yunaConfig["ai"]["repetition_penalty"],
                "top_a": 0,
                "typical": 1,
                "tfs": 0.8,
                "rep_pen_range": 512,
                "rep_pen_slope": 0,
                "sampler_order": [6, 5, 0, 2, 3, 1, 4],
                "memory": kanojo if kanojo is not None else "",
                "trim_stop": True,
                "genkey": "KCPP9126",
                "mirostat": 2,
                "mirostat_tau": 4,
                "mirostat_eta": 0.3,
                "dynatemp_range": 0,
                "dynatemp_exponent": 1,
                "smoothing_factor": 0,
                "banned_tokens": [],
                "render_special": True,
                "quiet": True,
                "stop_sequence": yunaConfig["ai"]["stop"],
                "use_default_badwordsids": False,
                "bypass_eos": False,
                "prompt": final_prompt,
            }
            payload = {**common_payload, **specific}
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
                    else:
                        return ''
            else:
                return ''
        else:
            return ''

    def load_audio_model(self):
        self.yunaListen = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            torch_dtype=torch.float32,
            device="mps" if torch.backends.mps.is_available() else "cpu",
            model_kwargs={"attn_implementation": "sdpa"},
        )

    def load_voice_model(self):
        if self.config["server"]["yuna_audio_mode"] == "hanasu":
            self.voice_model = load_model_hanasu(
                config_path=self.config['server']['voice_default_model'][0],
                model_path=self.config['server']['voice_default_model'][1]
            )

    def load_text_model(self):
        mode = self.config["server"].get("yuna_text_mode")
        if mode == "llamacpp":
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
        elif mode == "mlx": self.text_model, self.tokenizer =  load(self.config['server']['yuna_default_model'][0])
        elif mode == "mlxvlm": self.text_model, self.tokenizer = load(self.config['server']['yuna_default_model'][0])

    def load_kokoro_model(self, config, model_path): print("Kokoro is not available in this environment.")

    def export_audio(self, input_file, output_filename): AudioSegment.from_file(input_file).export(output_filename, format="mp3")
    def transcribe_audio(self, audio_file): return self.yunaListen(audio_file, chunk_length_s=30, batch_size=60, return_timestamps=False)['text']
    def speak_text(self, text, output_filename=None):
        output_filename = f"static/audio/{uuid.uuid4()}.mp3"
        mode = self.config['server']['yuna_audio_mode']

        if mode == 'siri':
            temp = "temp.aiff"
            os.system(f'say -o {temp} {repr(text)}')
            self.export_audio(temp, output_filename)
            os.remove(temp)
        elif mode == "siri-pv":
            temp = "static/audio/audio.aiff"
            voice_model = self.config['server']['voice_default_model'][0]
            os.system(f'say -v {voice_model} -o {temp} {repr(text)}')
            self.export_audio(temp, output_filename)
        elif mode == "hanasu":
            result = inference_hanasu(
                model=self.voice_model,
                text=text,
                noise_scale=0.2,
                noise_scale_w=1.0,
                length_scale=1.0,
                device="mps",
                stream=False,
            )

            write(data=result, rate=48000, filename="static/audio/temp.wav")
            return "static/audio/temp.wav"

    def processTextFile(self, text_file, question):
        from aiflow.utils import calculate_similarity
        from aiflow.models.kokoro.model import EmbeddingGenerator
        import re

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

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
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                result.append(sentences[-1].strip())
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

    def web_search(self, search_query):
        answer, search_results, image_urls = search_web(search_query)
        return {'answer': answer, 'results': search_results, 'images': image_urls}

    def blog_email(self, email, subject, message): pass

    def start(self):
        if self.config["ai"]["mind"]: self.load_text_model()
        if self.config["ai"]["audio"]: self.load_audio_model()
        if self.config["ai"]["hanasu"]: self.load_voice_model()
        if self.config["ai"]["kokoro"]: self.load_kokoro_model()