import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from datasets import load_dataset
from io import StringIO
from peft import PeftModel

# Conditional loading for Google Colab
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Conditional loading for unsloth and related modules
try:
    from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
except ImportError:
    pass

# New class with model creation, LoRA setup, dataset preparation and training
class YunaLLMTrainer:
    def __init__(self, model_path="yukiarimo/yuna-ai-v4", dataset_file="data.jsonl", output_dir="newmodel-output", max_seq_length=16384, load_in_4bit=True, device_map={"": 0}):
        self.model_path = model_path
        self.dataset_file = dataset_file
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.tokenizer = None
        self.model = None
        self.base_model = None
        self.trainer = None

        if IN_COLAB:
            drive.mount('/content/drive')

    def create_model(self, dtype=None):
        # Create model and tokenizer using FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=self.max_seq_length,
            dtype=dtype,
            load_in_4bit=self.load_in_4bit
        )
        print("Model and tokenizer created.")

    def setup_lora(self, r=128, lora_alpha=128, target_modules=None, lora_dropout=0.01, bias="all", use_gradient_checkpointing="unsloth", random_state=42, use_rslora=True, loftq_config=None):
        # Setup LoRA for the model
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head"]
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )
        print("LoRA setup complete.")

    def prepare_dataset(self):
        dataset = load_dataset("json", data_files=self.dataset_file)["train"]
        dataset = dataset.map(lambda examples: {"text": examples["text"]})
        print("Dataset prepared.")
        return dataset

    def train_model(self, training_args_kwargs=None):
        if training_args_kwargs is None:
            training_args_kwargs = {
                "output_dir": self.output_dir,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "max_steps": 300,
                "warmup_steps": 20,
                "warmup_ratio": 0.1,
                "learning_rate": 1e-6,
                "embedding_learning_rate": 1e-7,
                "bf16": True,
                "logging_steps": 1,
                "optim": "adamw_hf",
                "weight_decay": 0.01,
                "lr_scheduler_type": "constant",
                "seed": 42,
                "max_grad_norm": 1.0,
                "group_by_length": True,
                "report_to": "tensorboard",
                "save_steps": 10,
                "gradient_checkpointing": True,
            }
        dataset = self.prepare_dataset()
        training_args = UnslothTrainingArguments(**training_args_kwargs)
        self.trainer = UnslothTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            packing=False,
            args=training_args,
        )
        trainer_stats = self.trainer.train()
        print("Training complete.")
        return trainer_stats

    def merge_and_save_model(self, merged_model_dir="model-merged", checkpoint_dir="checkpoint-20"):
        # Load the base model in FP16
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map=self.device_map
        )
        # Load the PEFT model from checkpoint (example uses checkpoint-20)
        peft_model = PeftModel.from_pretrained(self.base_model, os.path.join(self.output_dir, checkpoint_dir))
        merged_model = peft_model.merge_and_unload()
        # Load and configure tokenizer from the checkpoint directory
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.output_dir, checkpoint_dir), trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        merged_model.save_pretrained(merged_model_dir)
        self.tokenizer.save_pretrained(merged_model_dir)
        print("Merged model and tokenizer saved.")
        return merged_model_dir

    def inference_setup(self, inference_model_path, max_seq_length=2048, dtype=None):
        # Load model and tokenizer for inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=inference_model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("Inference setup complete.")

    def generate_text(self, text, max_new_tokens=128):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        text_stream = StringIO()
        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=max_new_tokens)
        output = text_stream.getvalue()
        print("Text generation complete.")
        return output

    def create_and_save_new_model(self, base_model_name="yukiarimo/yuna-ai-v3", new_model_dir="yuna-ai-v4-base", temperature=0.7, additional_tokens=None):
        # Reload base model in FP16 and update configuration
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        base_model.generation_config.temperature = temperature
        # Load tokenizer for the base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        tokenizer.padding_side = "right"
        # Add any additional tokens if provided
        if additional_tokens:
            tokenizer.add_tokens(additional_tokens)
            base_model.resize_token_embeddings(len(tokenizer))
        base_model.save_pretrained(new_model_dir)
        tokenizer.save_pretrained(new_model_dir)
        print("New model created and saved.")
        return new_model_dir

    def huggingface_upload(self, repo="yukiarimo/yuna-ai-v4-atomic-trained", model_dir="yuna-atomic-pack3"):
        print("Starting Hugging Face CLI login...")
        if os.system("huggingface-cli whoami") == 0:
            os.system("huggingface-cli login")
        print("Login complete.")
        print("Uploading model directory '{}' to repository '{}'...".format(model_dir, repo))
        os.system("huggingface-cli upload {} {} .".format(repo, model_dir))
        print("Upload complete.")

    def create_atomic_model(self, atomic_model_dir="yuna-atomic-pack3", atomic_model_name="yukiarimo/yuna-ai-v4-atomic-trained"):
        """
        !git clone https://github.com/arcee-ai/mergekit.git
        %cd mergekit
        !pip install -e .
        !mergekit-yaml yuna.yaml ./yuna_new_model --cuda --lazy-unpickle --allow-crimes --clone-tensors
        """

        print("Atomic model creation is not yet implemented.")