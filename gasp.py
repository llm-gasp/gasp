import wandb
wandb.init(mode="disabled")

import argparse
from helper.blackbox import BlackBox
from helper.config import Config
from helper.inference import Inference
from helper.evaluator import Evaluator
from helper.tester import Tester
from helper.lbo import LBO
from helper.orpo import ORPO
from helper.suffixllm import SuffixLLM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pre-train')

    args = parser.parse_args()
    return args

def train(config, suffix_llm, evaluator, blackbox, orpo):
    suffix_llm.train_model()

    # Once trained, we need to set it to eval mode.
    suffix_llm.setup_inference()

    # Initialize necessary classes for ORPO
    lbo = LBO(config.data_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
    inference = Inference(config.data_cfg, suffix_llm, lbo)

    # Generate the data via LBO
    inference.align_lbo(suffix_llm.retraining_data)
    print("Pre-training complete!")

    # Train the ORPO model
    orpo.train(config.data_cfg["infer_save"])
    print("Training complete!")

def test(config, suffix_llm, evaluator, blackbox, orpo):
    # Load the ORPO model for inference.
    suffix_llm.load_orpo_model(config.blackbox_cfg["black_box_model"]["model"])

    # Initialize necessary classes for ORPO (if needed)
    lbo = LBO(config.data_cfg, suffix_llm.model, suffix_llm.tokenizer, blackbox, evaluator)
    inference = Inference(config.data_cfg, suffix_llm, lbo)
    
    # Begin evaluation.
    tester = Tester(config.data_cfg, suffix_llm, inference, orpo)
    tester.evaluate()
    print("Evaluation complete!")

def main():
    args = get_args()
    print("Performing task: ", args.task)

    # Initialize all necessary classes
    config = Config()
    suffix_llm = SuffixLLM(config.suffix_cfg, config.data_cfg)
    evaluator = Evaluator(config.evaluator_cfg)
    blackbox = BlackBox(config.blackbox_cfg)
    
    # Supplementary classes (ORPO)
    orpo = ORPO(config.suffix_cfg, suffix_llm, config.blackbox_cfg["black_box_model"]["model"])

    if args.task == 'all':
        train(config, suffix_llm, evaluator, blackbox, orpo)
        test(config, suffix_llm, evaluator, blackbox, orpo)

    elif args.task == 'train':
        train(config, suffix_llm, evaluator, blackbox, orpo)

    elif args.task == 'eval':
        test(config, suffix_llm, evaluator, blackbox, orpo)
        
    else:
        raise ValueError("Invalid task argument. Please use 'train', 'eval', or 'all'.")

# Run.
main()