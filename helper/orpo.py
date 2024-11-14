import pandas as pd
from datasets import Dataset as HFDataset
from trl import ORPOConfig, ORPOTrainer
from peft import (
    LoraConfig,
    get_peft_model,
)

class ORPO:
    def __init__(self, config, suffix_llm, blackbox_name):
        self.config = config
        self.suffix_llm = suffix_llm
        self.blackbox_name = blackbox_name

        self.seed = self.config["seed"]
        self.beta = self.config["orpo-training"]["beta"]
        self.num_epochs = self.config["orpo-training"]["num_train_epochs"]
        self.warmup_steps = self.config["orpo-training"]["warmup_steps"]
        self.weight_decay = self.config["orpo-training"]["weight_decay"]
        self.learning_rate = self.config["orpo-training"]["learning_rate"]
        self.logging_steps = self.config["orpo-training"]["logging_steps"]
        self.logging_dir = self.config["orpo-training"]["logging_dir"]
        self.output_dir = self.config["orpo-training"]["output_dir"]
        self.batch_size = self.config["orpo-training"]["batch_size"]

        self.lora_r = self.config["orpo-training"]["lora"]["r"]
        self.lora_alpha = self.config["orpo-training"]["lora"]["lora_alpha"]
        self.lora_dropout = self.config["orpo-training"]["lora"]["lora_dropout"]
        self.target_modules = self.config["orpo-training"]["lora"]["target_modules"]
        self.bias = self.config["orpo-training"]["lora"]["bias"]
        print("Class: ORPO Initialized")
        
    def load_models(self):
        self.suffix_llm.setup_inference()

        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type="CAUSAL_LM"
        )

        self.suffix_llm.model = get_peft_model(self.suffix_llm.model, config)

    def train(self, dataset_path):
        self.load_models()
        self.dataset = pd.read_csv(dataset_path)
        
        # Get prompt, chosen, rejected
        self.prompt = self.dataset["prompt"]
        self.chosen = self.dataset["chosen"]
        self.rejected = self.dataset["rejected"]

        # Get HFDataset from dictionary
        dataset = HFDataset.from_dict({
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected
        })

        orpo_config = ORPOConfig(
            beta=self.beta,
            num_train_epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            logging_dir=self.logging_dir,
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=1,
            bf16=True,
            save_strategy='epoch'
        )

        trainer = ORPOTrainer(
            model=self.suffix_llm.model,
            args=orpo_config,
            train_dataset=dataset,
            tokenizer=self.suffix_llm.tokenizer
        )

        print("[ORPO] Training Started")
        trainer.train()

        self.suffix_llm.model.save_pretrained(f"./models/{self.blackbox_name}_orpo")
        self.suffix_llm.tokenizer.save_pretrained(f"./models/{self.blackbox_name}_orpo")
        print("[ORPO] Training Completed")



        