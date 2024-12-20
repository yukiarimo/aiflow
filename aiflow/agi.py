import io
import json
import os
import uuid
import torch
import requests
import soundfile as sf
from pydub import AudioSegment, silence
from aiflow.helper import get_config, clearText, search_web, get_html, get_transcript
from transformers import pipeline
from tqdm import tqdm

def load_conditional_imports(config):
    mode = config["server"].get("yuna_text_mode")
    if config["ai"].get("agi"):
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_community.llms import LlamaCpp
        globals().update({
            'TextLoader': TextLoader,
            'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
            'Chroma': Chroma,
            'HuggingFaceEmbeddings': HuggingFaceEmbeddings,
            'RetrievalQA': RetrievalQA,
            'LlamaCpp': LlamaCpp
        })
    elif mode == "native":
        from llama_cpp import Llama
        globals().update({'Llama': Llama})

    # Audio imports
    if config['ai'].get('audio'):
        from transformers import pipeline
        globals().update({'pipeline': pipeline})

    # Voice synthesis imports
    if config['server'].get('yuna_audio_mode') == "11labs":
        from elevenlabs import VoiceSettings
        from elevenlabs.client import ElevenLabs
        globals().update({'VoiceSettings': VoiceSettings, 'ElevenLabs': ElevenLabs})
    elif config['server'].get('yuna_audio_mode') == "native":
        from gpt_sovits_python import TTS, TTS_Config
        globals().update({'TTS': TTS, 'TTS_Config': TTS_Config})
    elif config['server'].get('yuna_audio_mode') == "11labs":
        from elevenlabs import VoiceSettings
        from elevenlabs.client import ElevenLabs
        globals().update({'VoiceSettings': VoiceSettings, 'ElevenLabs': ElevenLabs})

class AGIWorker:
    def __init__(self, config=None):
        self.config = get_config() if config is None else config
        self.text_model = None
        self.image_model = None
        self.voice_model = None
        self.audio_model = None
        load_conditional_imports(self.config)

    def generate_text(self, text=None, kanojo=None, chat_history=None, useHistory=True, yunaConfig=None, stream=False):
        self.config = yunaConfig or self.config
        response = ''
    
        mode = self.config["server"]["yuna_text_mode"]
        if mode in {"native", "lmstudio"}:
            final_prompt = self.construct_prompt(text, kanojo, chat_history, useHistory, yunaConfig, mode)
            if mode == "native":
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
                return (chunk['choices'][0]['text'] for chunk in response) if stream else clearText(str(response['choices'][0]['text']))
            elif mode == "lmstudio":
                dataSendAPI = {
                    "model": f"{self.config['server']['yuna_default_model']}",
                    "messages": self.get_history_text(chat_history, text, useHistory, yunaConfig),
                    "temperature": yunaConfig["ai"]["temperature"],
                    "max_tokens": -1,
                    "stop": yunaConfig["ai"]["stop"],
                    "top_p": yunaConfig["ai"]["top_p"],
                    "top_k": yunaConfig["ai"]["top_k"],
                    "min_p": 0,
                    "presence_penalty": 0,
                    "frequency_penalty": 0,
                    "logit_bias": {},
                    "repeat_penalty": yunaConfig["ai"]["repetition_penalty"],
                    "seed": yunaConfig["ai"]["seed"]
                }
                response = requests.post("http://localhost:1234/v1/chat/completions", headers={"Content-Type": "application/json"}, json=dataSendAPI, stream=stream)
                if response.status_code == 200:
                    resp = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
                    return ''.join(resp) if stream else clearText(resp)
        elif mode == "koboldcpp":
            return self.handle_koboldcpp_mode(text, kanojo, chat_history, yunaConfig, stream)
        return response

    def construct_prompt(self, text, kanojo, chat_history, useHistory, yunaConfig, mode):
        if kanojo:
            tokens = self.text_model.tokenize(kanojo.encode('utf-8'))
            history_text = self.get_history_text(chat_history, text, useHistory, yunaConfig)
            max_tokens = yunaConfig["ai"]["context_length"] - len(tokens) - yunaConfig["ai"]["max_new_tokens"]
            tokens_history = self.text_model.tokenize(history_text.encode('utf-8'))
            if len(tokens_history) > max_tokens:
                tokens_history = tokens_history[-max_tokens:]
                history_text = self.text_model.detokenize(tokens_history).decode('utf-8')
            return self.text_model.detokenize(tokens).decode('utf-8') + history_text
        return text

    def handle_koboldcpp_mode(self, text, kanojo, chat_history, yunaConfig, stream):
        messages = self.get_history_text(chat_history, text, useHistory=True, yunaConfig=yunaConfig)
        formatted_prompt = f"{kanojo}{messages}" if kanojo else messages
        call_api = {
            "n": 1,
            "max_context_length": yunaConfig["ai"]["context_length"],
            "max_length": yunaConfig["ai"]["max_new_tokens"],
            "rep_pen": yunaConfig["ai"]["repetition_penalty"],
            "temperature": yunaConfig["ai"]["temperature"],
            "top_p": yunaConfig["ai"]["top_p"],
            "top_k": yunaConfig["ai"]["top_k"],
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
            "min_p": 0.1,
            "dynatemp_range": 0,
            "dynatemp_exponent": 1,
            "smoothing_factor": 0,
            "banned_tokens": [],
            "render_special": True,
            "presence_penalty": 0,
            "logit_bias": {},
            "prompt": formatted_prompt,
            "quiet": True,
            "stop_sequence": yunaConfig["ai"]["stop"],
            "use_default_badwordsids": False,
            "bypass_eos": False,
        }
        url = "http://localhost:5001/api/extra/generate/stream/" if stream else "http://localhost:5001/api/v1/generate/"
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=call_api, stream=stream)
        
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

    def get_history_text(self, chat_history, text, useHistory, yunaConfig):
        user_name = yunaConfig["ai"]["names"][0].lower()
        assistant_name = yunaConfig["ai"]["names"][1].lower()
        
        history = ''.join([
            f"<{user_name}>{item.get('message', '')}</{user_name}>\n" if item.get('name', '').lower() == user_name else f"<{assistant_name}>{item.get('message', '')}</{assistant_name}>\n"
            for item in chat_history
        ]) if useHistory else ''
        
        return f"{history}<{user_name}>{text}</{user_name}>\n<{assistant_name}>"

    def load_image_model(self):
        if self.config["ai"]["miru"]:
            if self.config["server"]["miru_default_model"] == "":
                raise ValueError("No default model set for miru")
            elif not os.path.exists(self.config["server"]["miru_default_model"]):
                raise FileNotFoundError(f"Model {self.config['server']['miru_default_model']} not found")
            elif self.config["server"]["miru_model_type"] == "moondream":
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import MoondreamChatHandler

                self.image_model = Llama(
                    model_path=self.config['server']['miru_default_model'],
                    chat_handler=MoondreamChatHandler(clip_model_path=self.config['server']['eyes_default_model']),
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
        if image_path is None:
            raise ValueError("No image path provided")
        if prompt is None:
            raise ValueError("No prompt provided")
        if self.image_model is None:
            raise ValueError("No image model loaded")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found")
        if self.config["server"]["miru_model_type"] == "moondream":
            abs_image_path = os.path.join(os.getcwd(), image_path)
            result = self.image_model.create_chat_completion(messages=[
                {"role": "system", "content": "You are an assistant who perfectly describes images and answers questions about them."},
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"file://{abs_image_path}"}}]}
            ])
            answer = clearText(result['choices'][0]['message']['content'])
            return [answer, image_path]

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

    def export_audio(self, input_file, output_filename):
        audio = AudioSegment.from_file(input_file)
        audio.export(output_filename, format="mp3")

    def transcribe_audio(self, audio_file): return self.yunaListen(audio_file, chunk_length_s=30, batch_size=60, return_timestamps=False)['text']

    def speak_text(self, text, project=None, chapter=None, paragraph=None, mode=None, reference_audio=None, output_filename="audio.mp3"):
        reference_audio = reference_audio or self.config['server']['yuna_reference_audio']
        mode = mode or self.config['server']['yuna_audio_mode']

        if mode == "native":
            # Initialize common parameters
            self.tts_params.update({
                "text": text,
                "ref_audio_path": reference_audio,
            })

        # Create directory structure if it doesn't exist
        if project and chapter:
            audio_dir = f"static/audio/audiobooks/{project}/{chapter}"
            os.makedirs(audio_dir, exist_ok=True)
            
            if paragraph:
                output_filename = f"{audio_dir}/paragraph_{paragraph}.mp3"
            else:
                output_filename = f"{audio_dir}/full_chapter.mp3"
        else:
            output_filename = f"static/audio/{uuid.uuid4()}.mp3"

        if mode == 'siri':
            temp_aiff = "temp.aiff"
            os.system(f'say -o {temp_aiff} {repr(text)}')
            self.export_audio(temp_aiff, output_filename)
            os.remove(temp_aiff)

        elif mode == "siri-pv":
            temp_aiff = "static/audio/audio.aiff"
            os.system(f'say -v {reference_audio} -o {temp_aiff} {repr(text)}')
            self.export_audio(temp_aiff, output_filename)

        elif mode == "native":
            with torch.no_grad():
                tts_generator = self.tts_pipeline.run(self.tts_params)
                sr, audio_data = next(tts_generator)
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sr, format='WAV')
                buffer.seek(0)
                self.export_audio(buffer, output_filename)

        elif mode == "11labs":
            client = ElevenLabs(api_key=self.config['security']['11labs_key'])
            audio_bytes = b''.join(client.generate(
                text=text,
                voice="Yuna Upgrade Use",
                voice_settings=VoiceSettings(
                    stability=0.40,
                    similarity_boost=1.00,
                    style=0.00,
                    use_speaker_boost=True
                ),
                model="eleven_multilingual_v2",
                optimize_streaming_latency=0,
                stream=False,
                output_format="mp3_44100_192", # pcm_44100
            ))

            # Save the generated audio to a file
            with open(output_filename, "wb") as f:
                f.write(audio_bytes)
            #self.export_audio(audio_bytes, output_filename)

        # Check if audio file exists
        if os.path.exists(output_filename):
            print(f"Audio file {output_filename} created")
            return '/' + output_filename
        else:
            print(f"Failed to create audio file {output_filename}")
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
        if self.config["ai"]["voice"]:
            self.load_voice_model()

class AudioDataWorker:
    def __init__(self, device="mps", model="openai/whisper-tiny"):
        self.device = device
        self.model = model
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            torch_dtype=torch.float32,
            device=self.device,
            model_kwargs={"attn_implementation": "sdpa"},
        )

    def combine_audio_files(self, input_folder, output_file, silence_duration=1000):
        audio_files = sorted(
            [f for f in os.listdir(input_folder) if f.endswith('.wav')],
            key=lambda x: int(x.split('.')[0])
        )
        combined = AudioSegment.empty()
        one_sec_silence = AudioSegment.silent(duration=silence_duration)
        for i, file in enumerate(audio_files):
            audio = AudioSegment.from_wav(os.path.join(input_folder, file))
            combined += audio
            if i < len(audio_files) - 1:
                combined += one_sec_silence
        combined.export(output_file, format="wav")
        print(f"Combined audio saved as {output_file}")

    def split_audio_into_chunks(self, input_file, output_dir, min_silence_len=400, silence_thresh=-40,
                                min_chunk_length=4000, max_chunk_length=20000):
        os.makedirs(output_dir, exist_ok=True)
        audio = AudioSegment.from_wav(input_file)
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=500
        )
        
        def process_chunks(chunks, min_len, max_len):
            processed, current = [], AudioSegment.empty()
            for chunk in chunks:
                current += chunk
                if len(current) >= min_len:
                    if len(current) > max_len:
                        processed.append(current[:max_len])
                        current = current[max_len:]
                    else:
                        processed.append(current)
                        current = AudioSegment.empty()
            if len(current) > 0:
                processed.append(current)
            return processed

        final_chunks = process_chunks(chunks, min_chunk_length, max_chunk_length)
        for i, chunk in enumerate(final_chunks):
            chunk_filename = os.path.join(output_dir, f"chunk_{i+1}.wav")
            chunk.export(chunk_filename, format="wav")
            print(f"Exported {chunk_filename}")
        print("Splitting completed successfully!")

    def calculate_total_duration(self, directory):
        wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
        total_ms = 0
        print("Calculating total duration...")
        for wav_file in tqdm(wav_files, desc="Processing files"):
            try:
                audio = AudioSegment.from_wav(os.path.join(directory, wav_file))
                total_ms += len(audio)
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
        total_seconds = total_ms / 1000
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTotal duration:\nHours: {int(hours)}\nMinutes: {int(minutes)}\nSeconds: {int(seconds)}")
        print(f"Total files processed: {len(wav_files)}")

    def transcribe_audio(self, audio_file, chunk_length_s=30, batch_size=24, return_timestamps=False):
        result = self.asr_pipeline(
            audio_file,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
        )
        return result

    def transcribe_directory(self, directory_path, output_file, chunk_length_s=30, batch_size=24, return_timestamps=False):
        with open(output_file, 'w') as f:
            for filename in os.listdir(directory_path):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(directory_path, filename)
                    transcription = self.transcribe_audio(audio_path, chunk_length_s, batch_size, return_timestamps)
                    f.write(f"{audio_path}|{transcription}\n")
                    print(f"Transcribed {audio_path}")

    def delete_unlisted_files(self, metadata_file, directory):
        with open(metadata_file, 'r') as file:
            lines = file.readlines()
        metadata_files = {line.strip().split('|')[0] for line in lines}
        all_files = {os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')}
        files_to_delete = all_files - metadata_files
        for file_path in files_to_delete:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        print("Deletion of unlisted files completed successfully!")

    def rename_files(self, metadata_file, directory, start_number=2565):
        with open(metadata_file, 'r') as file:
            lines = file.readlines()
        metadata = [line.strip().split('|') for line in lines]
        for i, data in enumerate(metadata):
            old_path = data[0]
            new_name = f"{start_number + i}.wav"
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)
            data[0] = new_path
            print(f"Renamed {old_path} to {new_path}")
        with open(metadata_file, 'w') as file:
            for data in metadata:
                file.write('|'.join(data) + '\n')
        print("Renaming and metadata update completed successfully!")

    def start_ui(self):
        transcript = input("Enter the path to the transcript file: ")

        from aiflow.AudioDataWorkerUI import main
        main(transcript)

class TextDataWorker:
    pass