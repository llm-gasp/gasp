import json
from huggingface_hub import login
import os

class Config:
    def __init__(self):
        self.suffix_cfg = self.load_config("./config/suffix_llm.json")
        self.evaluator_cfg = self.load_config("./config/evaluator.json")
        self.data_cfg = self.load_config("./config/data.json")
        self.blackbox_cfg = self.load_config("./config/blackbox.json")
        
        self.initialize_api_keys()
        
        print("Class: Config Initialized")

    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        
        return config
    
    def initialize_api_keys(self):
        self.hf_api_key = self.data_cfg["HF_API_KEY"]
        self.openai_api_key = self.data_cfg["OPENAI_API_KEY"]
        
        if self.hf_api_key == "[MODIFY THIS]" or self.hf_api_key == "":
            print("HuggingFace API Key is not set. This may cause errors with gated models.")
        else:
            login(token=self.hf_api_key)
        
        if self.openai_api_key == "[MODIFY THIS]" or self.openai_api_key == "":
            print("OpenAI API Key is not set. This may cause errors if using OpenAI models.")
        else:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
                
        