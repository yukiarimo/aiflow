import io
import json
import os
import uuid
import torch
import requests
import soundfile as sf
from aiflow.helper import get_config, clearText, search_web, get_html, get_transcript
from transformers import pipeline

def load_conditional_imports(config):
    imports = {
        'agi': {
            'from': ['langchain_community.document_loaders', 'langchain.text_splitter', 'langchain_community.vectorstores',
                    'langchain_huggingface', 'langchain.chains', 'langchain_community.llms'],
            'import': ['TextLoader', 'RecursiveCharacterTextSplitter', 'Chroma', 'HuggingFaceEmbeddings', 'RetrievalQA', 'LlamaCpp']
        },
        'native': {'from': ['llama_cpp'], 'import': ['Llama']},
        'mlx': {'from': ['mlx_lm'], 'import': ['generate', 'load']},
        'audio': {'from': ['transformers'], 'import': ['pipeline']},
        '11labs': {'from': ['elevenlabs', 'elevenlabs.client'], 'import': ['VoiceSettings', 'ElevenLabs']},
        'native_voice': {'from': ['gpt_sovits_python'], 'import': ['TTS', 'TTS_Config']}
    }

    mode = config["server"].get("yuna_text_mode")
    to_import = []

    if config["ai"].get("agi"):
        to_import.append('agi')
    elif mode in ["native", "mlx"]:
        to_import.append(mode)
    
    if config['ai'].get('audio'):
        to_import.append('audio')
    
    audio_mode = config['server'].get('yuna_audio_mode')
    if audio_mode in ["11labs", "native"]:
        to_import.append('11labs' if audio_mode == "11labs" else 'native_voice')

    for imp in to_import:
        for module, items in zip(imports[imp]['from'], imports[imp]['import']):
            exec(f"from {module} import {items}")
            globals()[items] = eval(items)

class AGIWorker:
    def __init__(self, config=None):
        self.config = get_config() if config is None else config
        self.text_model = None
        self.tokenizer = None
        self.image_model = None
        self.voice_model = None
        self.audio_model = None
        load_conditional_imports(self.config)

    def get_history_text(self, chat_history, text, useHistory, yunaConfig):
        user, asst = yunaConfig["ai"]["names"][0].lower(), yunaConfig["ai"]["names"][1].lower()
        history = ''.join([f"<{user if m['name'].lower() == user else asst}>{m['text']}</{user if m['name'].lower() == user else asst}>\n" 
                          for m in (chat_history or [])] if useHistory else '')
        current = text.get('text') if isinstance(text, dict) else text
        final = f"{history}<{user}>{current}</{user}>\n<{asst}>"
        return final

    def generate_text(self, text=None, kanojo=None, chat_history=None, useHistory=True, yunaConfig=None, stream=False):
        self.config = yunaConfig or self.config
        mode = self.config["server"]["yuna_text_mode"]
        # Build the full prompt (including history).
        final_prompt = self.get_history_text(chat_history, text, useHistory, yunaConfig)

        if mode == "llamacpp":
            response = self.text_model(
                final_prompt,
                stream=stream,
                top_k=yunaConfig["ai"]["top_k"],
                top_p=yunaConfig["ai"]["top_p"],
                temperature=yunaConfig["ai"]["temperature"],
                repeat_penalty=yunaConfig["ai"]["repetition_penalty"],
                max_tokens=yunaConfig["ai"]["max_new_tokens"],
                stop=yunaConfig["ai"]["stop"],
            )
            if stream:
                return (chunk['choices'][0]['text'] for chunk in response)
            else:
                return clearText(str(response['choices'][0]['text']))
        elif mode == "mlx":
            kwargs = {
                "max_tokens": 100,
                "prefill_step_size": 2048,
                "kv_group_size": 16,
                "quantized_kv_start": 0,
            }

            text = generate(model=self.text_model, tokenizer=self.tokenizer, prompt="The meaning of life is", verbose=True, **kwargs)
            return text
        elif mode in {"lmstudio", "koboldcpp"}:
            # Build payload dynamically using base/common sub-dicts.
            common_payload = {
                "temperature": yunaConfig["ai"]["temperature"],
                "top_p": yunaConfig["ai"]["top_p"],
                "top_k": yunaConfig["ai"]["top_k"],
                "min_p": 0.2,
                "logit_bias": {},
                "presence_penalty": 0,
            }
            if mode == "lmstudio":
                specific = {
                    "model": self.config["server"]["yuna_default_model"],
                    "max_tokens": -1,
                    "stop": yunaConfig["ai"]["stop"],
                    "frequency_penalty": 0,
                    "repeat_penalty": yunaConfig["ai"]["repetition_penalty"],
                    "seed": yunaConfig["ai"]["seed"],
                    "messages": final_prompt,
                }
                payload = {**common_payload, **specific}
                url = "http://localhost:1234/v1/chat/completions"
            elif mode == "koboldcpp":
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
                    "memory": kanojo,
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
                if mode == "lmstudio":
                    resp = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                    return (''.join(resp) if stream else clearText(resp))
                else:  # koboldcpp
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

    def load_image_model(self):
        if self.config["ai"]["miru"]:
            if self.config["server"]["miru_default_model"] == "":
                raise ValueError("No default model set for miru")
            elif not os.path.exists(self.config["server"]["miru_default_model"]):
                raise FileNotFoundError(f"Model {self.config['server']['miru_default_model'][0]} not found")
            elif self.config["server"]["yuna_miru_mode"] == "moondream":
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import MoondreamChatHandler

                self.image_model = Llama(
                    model_path=self.config['server']['miru_default_model'][0],
                    chat_handler=MoondreamChatHandler(clip_model_path=self.config['server']['miru_default_model'][1]),
                    n_ctx=4096,
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
        else:
            raise ValueError("Miru is not enabled")

    def capture_image(self, image_path=None, prompt=None):
        if not all([image_path, prompt, self.image_model]) or not os.path.exists(image_path):
            raise ValueError("Missing required inputs or image not found")
            
        if self.config["server"]["yuna_miru_mode"] == "moondream":
            result = self.image_model.create_chat_completion(messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images and answers questions about them."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"file://{os.path.join(os.getcwd(), image_path)}"}}]}
            ])
            return [clearText(result['choices'][0]['message']['content']), image_path]

    def load_audio_model(self):
        self.yunaListen = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            torch_dtype=torch.float32,
            device="mps" if torch.backends.mps.is_available() else "cpu",
            model_kwargs={"attn_implementation": "sdpa"},
        )

    def load_voice_model(self):
        if self.config["server"]["yuna_audio_mode"] == "native":
            soviets_configs = {
                "default": {
                    "device": "cpu",
                    "is_half": False,
                    "t2s_weights_path": f"lib/utils/models/agi/voice/{self.config['server']['voice_default_model']}/{self.config['server']['voice_model_config'][0]}",
                    "vits_weights_path": f"lib/utils/models/agi/voice/{self.config['server']['voice_default_model']}/{self.config['server']['voice_model_config'][1]}",
                    "cnhuhbert_base_path": f"lib/utils/models/agi/voice/{self.config['server']['voice_default_model']}/chinese-hubert-base",
                    "bert_base_path": f"lib/utils/models/agi/voice/{self.config['server']['voice_default_model']}/chinese-roberta-wwm-ext-large"
                }
            }
            self.tts_config = TTS_Config(soviets_configs)
            self.tts_pipeline = TTS(self.tts_config)
            self.tts_params = {
                "text_lang": "en",
                "ref_audio_path": self.config['server']['yuna_reference_audio'],
                "prompt_text": "",
                "prompt_lang": "en",
                "top_k": 1,
                "top_p": 0.6,
                "temperature": 0.7,
                "text_split_method": "cut0",
                "batch_size": 1,
                "batch_threshold": 1.0,
                "split_bucket": True,
                "speed_factor": 1.0,
                "fragment_interval": 0.3,
                "seed": 1234,
                "media_type": "wav",
                "streaming_mode": False,
                "parallel_infer": True,
                "repetition_penalty": 1.25
            }

    def load_text_model(self):
            mode = self.config["server"].get("yuna_text_mode")
            if mode == "native":
                self.text_model = Llama(
                    model_path=f"lib/utils/models/yuna/{self.config['server']['yuna_default_model']}",
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
            elif mode == "mlx":
                from mlx_lm import load
                self.text_model, self.tokenizer =  load(f"lib/utils/models/yuna/{self.config['server']['yuna_default_model']}")

    def export_audio(self, input_file, output_filename):
        audio = AudioSegment.from_file(input_file)
        audio.export(output_filename, format="mp3")

    def transcribe_audio(self, audio_file): return self.yunaListen(audio_file, chunk_length_s=30, batch_size=60, return_timestamps=False)['text']

    def speak_text(self, text, output_filename="audio.mp3"):
        output_filename = f"static/audio/{uuid.uuid4()}.mp3"
        mode = self.config['server']['yuna_audio_mode']
        ref_audio = self.config['server']['yuna_reference_audio']

        try:
            if mode == 'siri':
                temp = "temp.aiff"
                os.system(f'say -o {temp} {repr(text)}')
                self.export_audio(temp, output_filename)
                os.remove(temp)
            elif mode == "siri-pv":
                temp = "static/audio/audio.aiff"
                os.system(f'say -v {ref_audio} -o {temp} {repr(text)}')
                self.export_audio(temp, output_filename)
            elif mode == "native":
                self.tts_params.update({"text": text, "ref_audio_path": ref_audio})
                with torch.no_grad():
                    sr, audio = next(self.tts_pipeline.run(self.tts_params))
                    buffer = io.BytesIO()
                    sf.write(buffer, audio, sr, format='WAV')
                    buffer.seek(0)
                    self.export_audio(buffer, output_filename)
            elif mode == "11labs":
                audio = b''.join(ElevenLabs(api_key=self.config['security']['11labs_key']).generate(
                    text=text, voice="Yuna Upgrade Use",
                    voice_settings=VoiceSettings(stability=0.40, similarity_boost=1.00, style=0.00, use_speaker_boost=True),
                    model="eleven_multilingual_v2", stream=False, output_format="mp3_44100_192"
                ))
                with open(output_filename, "wb") as f:
                    f.write(audio)

            return '/' + output_filename if os.path.exists(output_filename) else None
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def processTextFile(self, text_file, question, temperature):
            loader, splitter = TextLoader(text_file), RecursiveCharacterTextSplitter(chunk_size=200)
            docs = splitter.split_documents(loader.load())
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
            llm = LlamaCpp(model_path="lib/utils/models/yuna/yuna-ai-v3-q5_k_m.gguf", temperature=temperature, verbose=False)
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
            return qa.invoke(question).get('result', '')

    def web_search(self, search_query):
        answer, search_results, image_urls = search_web(search_query)
        return {
            'answer': answer,
            'results': search_results,
            'images': image_urls
        }

    def scrape_webpage(self, url):
        html_content = get_html(url)
        return html_content

    def get_youtube_transcript(self, url):
        transcript = get_transcript(url)
        return transcript

    def blog_email(self, email, subject, message):
        pass

    def start(self):
        if self.config["ai"]["mind"]:
            self.load_text_model()
        if self.config["ai"]["miru"]:
            self.load_image_model()
        if self.config["ai"]["audio"]:
            self.load_audio_model()
        if self.config["ai"]["hanasu"]:
            self.load_voice_model()