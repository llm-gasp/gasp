import torch
import os
import backoff 
import openai
from openai import OpenAI
from .logging import Logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

class BlackBox:
    def __init__(self, config):
        self.config = config        
        self.seed = self.config["seed"]
        self.blackbox_name = self.config["black_box_model"]["model"]
        self.blackbox_path = self.config["black_box_model"]["model_path"]
        self.provider = self.config["black_box_model"]["provider"]
        self.device = self.config["black_box_model"]["device"]
        
        self.temperature = self.config["black_box_params"]["temperature"]
        self.top_p = self.config["black_box_params"]["top_p"]
        self.max_length = self.config["black_box_params"]["max_length"]
        
        self.logger = Logging(self.config["black_box_model"]["logging_file"])
        print("Class: BlackBox Initialized")
        
        self.load_model()
        
    def load_model(self):
        if self.provider == "openai":
            self.client = OpenAI()
        elif self.provider == "huggingface": 
            self.model = AutoModelForCausalLM.from_pretrained(self.blackbox_path,
                                                            torch_dtype=torch.float16,
                                                            trust_remote_code=True).to(self.device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(self.blackbox_path,
                                                            trust_remote_code=True,
                                                            use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError("GASP only supports huggingface and openai as providers. Please check the config file.")
        
        print("[BlackBox] Model Loaded")

    def query(self, prompt):
        chat = [
            {"role": "system", "content": "You are a AI assistant that is designed to answer user's queries."},
            {"role": "user", "content": prompt}
        ]
        
        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(chat):
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=chat
            )
            
            return completion.choices[0].message.content
        
        if self.provider == "openai":
            llm_response = completions_with_backoff(chat)
        elif self.provider == "huggingface":
            formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted_chat, return_tensors='pt', add_special_tokens=False, padding=True)
            inputs = {key: tensor.to(self.model.device) for key, tensor in inputs.items()}
            
            outputs = self.model.generate(**inputs, max_length=self.max_length, temperature=self.temperature, top_p=self.top_p, do_sample=True)
            
            llm_response = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        else:
            raise ValueError("GASP only supports huggingface and openai as providers. Please check the config file.")
        
        self.logger.log(["PROMPT: " + prompt, "BLACKBOX-RESPONSE: " + llm_response])   
        return llm_response