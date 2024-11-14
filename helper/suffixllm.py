import os
import torch
import pandas as pd
from .loader import DatasetLoader
from .logging import Logging
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

class SuffixLLM:
    def __init__(self, config, dataset):
        # Training Directives
        self.config = config
        self.seed = self.config["seed"]
        self.model_name = self.config["model"]["model_name"]
        self.model_path = self.config["model"]["model_path"]
        self.device = self.config["model"]["device"]
        self.batch_size = self.config["model"]["batch_size"]
        self.num_epochs = self.config["model"]["num_train_epochs"]
        self.warmup_steps = self.config["model"]["warmup_steps"]
        self.weight_decay = self.config["model"]["weight_decay"]
        self.learning_rate = self.config["model"]["learning_rate"]
        self.logging_steps = self.config["model"]["logging_steps"]
        self.logging_dir = self.config["model"]["logging_dir"]
        self.output_dir = self.config["model"]["output_dir"]

        # LoRA Directives
        self.lora_r = self.config["model"]["lora"]["r"]
        self.lora_alpha = self.config["model"]["lora"]["lora_alpha"]
        self.lora_dropout = self.config["model"]["lora"]["lora_dropout"]
        self.target_modules = self.config["model"]["lora"]["target_modules"]
        self.bias = self.config["model"]["lora"]["bias"]

        # Inference Directives
        self.max_length = self.config["inference"]["max_length"]
        self.num_return_sequences = self.config["inference"]["num_return_sequences"]
        self.temperature = self.config["inference"]["temperature"]
        self.top_p = self.config["inference"]["top_p"]
        self.repetition_penalty = self.config["inference"]["repetition_penalty"]
        self.length_penalty = self.config["inference"]["length_penalty"]
        self.max_suffix_length = self.config["inference"]["max_suffix_length"]
        
        self.logger = Logging(self.config["model"]["suffix_logs"])
        self.dataset_name = dataset["dataset"]["name"]
        self.dataset_path = dataset["dataset"]["data_path"]
        self.SPLIT = dataset["dataset"]["split"]
        print("Class: SuffixLLM Initialized")

        self.TRAINING = self.check_if_trained()
        print("[SUFFIX-LLM] Model Trained: ", self.TRAINING)
    
    def load_dataset(self):
        self.data = pd.read_csv(self.dataset_path)
        
        # Shuffle the data
        self.data = self.data.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        self.training_data = self.data[:int(len(self.data)*self.SPLIT)]
        self.retraining_data = self.data[int(len(self.data)*self.SPLIT):]
        print("[SUFFIX-LLM] Dataset loaded; Training Data: ", len(self.training_data), " Retraining Data: ", len(self.retraining_data))
    
    def check_if_trained(self):
        # Check if model_name + "_finetuned" exists in the models directory
        if not os.path.exists(f"./models/{self.model_name}_finetuned"):
            return False
        return True
    
    def check_if_orpo_trained(self):
        # Check if model_name + "_orpo" exists in the models directory
        if not os.path.exists(f"./gasp-gpt3/models/{self.model_name}_orpo"):
            return False
        return True
    
    def train_model(self):
        if self.TRAINING == True:
            return
        
        self.load_dataset()
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                        torch_dtype=torch.float16,
                                                        trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                        trust_remote_code=True,
                                                        use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[SUFFIX-LLM] Model Loaded")
        
        self.dataset = DatasetLoader(self.training_data, self.tokenizer)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.dataset.data_collator)
                
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
                
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            bf16=True,
            auto_find_batch_size=True,
            overwrite_output_dir=True,
            save_strategy='epoch',
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            data_collator=self.dataset.data_collator,
        )
        
        self.model.config.use_cache = False

        print("[SUFFIX-LLM] Training Started")

        self.trainer.train()
        self.model.save_pretrained(f"./models/{self.model_name}_finetuned")
        self.tokenizer.save_pretrained(f"./models/{self.model_name}_finetuned")
        print("[SUFFIX-LLM] Training Completed")

    def setup_inference(self):
        # Remove the model and tokenizer from memory, if they exist
        if self.TRAINING == False or hasattr(self, 'model') == True:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

        self.load_dataset()
        
        # If an ORPO model is trained, load that instead
        if self.check_if_orpo_trained() == True:
            self.load_orpo_model(self.model_name)
            return

        self.model = AutoModelForCausalLM.from_pretrained(f"./models/{self.model_name}_finetuned",
                                                            torch_dtype=torch.float16,
                                                            trust_remote_code=True,
                                                            device_map="auto")
        # .to(self.device) -- Seems like ORPO hates this. OOM at all times.
        self.tokenizer = AutoTokenizer.from_pretrained(f"./models/{self.model_name}_finetuned",
                                                        trust_remote_code=True,
                                                        use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[SUFFIX-LLM] Inference Model Loaded")
        
    def load_orpo_model(self, blackbox_name):
        if hasattr(self, 'model') == True:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
        
        self.model = AutoModelForCausalLM.from_pretrained(f"./models/{blackbox_name}_orpo",
                                                            torch_dtype=torch.float16,
                                                            trust_remote_code=True,
                                                            device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(f"./models/{blackbox_name}_orpo",
                                                        trust_remote_code=True,
                                                        use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("[SUFFIX-LLM] ORPO Model Loaded")
    
    def generate_suffix(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            chat_completion = self.model.generate(
                **inputs,
                max_length=len(prompt) + self.max_length,
                num_return_sequences=self.num_return_sequences,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
            )
        
        generated_text = self.tokenizer.decode(chat_completion[0], skip_special_tokens=True)
        
        generated_text = generated_text.replace(prompt, "")
                
        # Split to get the suffixes, as per the delimiter '|', there will be one suffix between 2 |'s
        suffixes = []
        for i in generated_text.split('|'):
            suffixes.append(i.strip())

        # Remove if there are any empty suffixes and if the suffix is beyond the limit
        suffixes = [suffix for suffix in suffixes if suffix != "" and len(suffix.split()) < self.max_suffix_length]
        
        self.logger.log(["PROMPT: " + prompt, "SUFFIXES: " + str(suffixes)])
        return suffixes, generated_text