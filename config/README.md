# GASP - Configurations

We list all configuration options available in the config files: `blackbox.json`, `data.json`, `evaluator.json`, and `suffix_llm.json`.

- **`blackbox.json`**: Provides all configuration settings for the black-box LLM.
- **`data.json`**: Provides all configuration settings for data management and optimization.
- **`evaluator.json`**: Contains all configuration settings for the evaluation module.
- **`suffix_llm.json`**: Details configuration settings for suffix generation and model fine-tuning.

Please review all configuration settings with **"[MODIFY THIS]"** in the field values. Apart from the settings listed here, you need not modify others.

---

## `blackbox.json`

- **seed**: Sets the random seed for reproducibility. Default is `42`.

### `black_box_model`
- **model**: Specifies the model name to be used (e.g., `"mistral"`). Modify as needed.
- **model_path**: Local path to the model. Leave empty if not applicable.
- **provider**: Specifies the model provider. Options are `"huggingface"` or `"openai"`.
- **device**: Sets the device for computation, e.g., `"cuda:2"`.
- **logging_file**: Path for the log file that records black box operation logs.

### `black_box_params`[^1]
- **temperature**: Controls randomness in text generation. Higher values increase diversity.
- **max_length**: Maximum number of tokens to generate.
- **top_p**: Cumulative probability for top-p sampling, controlling the diversity of generated text.

---

## `data.json`

- **HF_API_KEY**: Hugging Face API key. Required if using Hugging Face models.
- **OPENAI_API_KEY**: OpenAI API key. Required if using OpenAI models.

### `lbo_params`[^2]
- **acquisition_function**: Specifies the acquisition function for Bayesian optimization. Options are `"EI"` (Expected Improvement), `"PI"` (Probability of Improvement), or `"LCB"` (Lower Confidence Bound).
- **n_calls**: Number of optimization calls.
- **seed**: Seed for reproducibility.
- **acq_optimizer**: Optimizer type for acquisition, such as `"sampling"`.
- **n_initial_points**: Initial points in the search space for optimization.
- **searches**: Number of search iterations.

### Additional Evaluation Parameters
- **eval_max_suffix_additions**: Maximum suffix additions allowed during evaluation.
- **orpo_finetune**: Boolean flag to enable or disable ORPO fine-tuning (during evaluation) -- recommended to be false; all experiments were ran on this setting.
- **orpo_num_prompts**: Number of prompts used for ORPO fine-tuning (during evaluation).

---

## `evaluator.json`

- **seed**: Sets the random seed for reproducibility. Default is `42`.
- **gaspeval_msg**: Path to the file containing the evaluation prompt for the GASP model.
- **strongreject_msg**: Path to the file containing the evaluation prompt for strong rejection cases.

### `evaluator_model`
- **model**: Name of the evaluator model to use (e.g., `"llama3.1-8b"`).
- **model_path**: Path to the local model. Leave empty if not applicable.
- **provider**: Specifies the model provider, either `"huggingface"` or `"openai"`.
- **device**: Sets the computation device, e.g., `"cuda:1"`.
- **logging_file**: Path for evaluator log file.

### `evaluator_params`[^1]
- **temperature**: Controls randomness in evaluation output.
- **max_length**: Maximum number of tokens to generate for evaluation.
- **top_p**: Controls diversity of generated text.

---

## `suffix_llm.json`

- **seed**: Sets the random seed for reproducibility. Default is `42`.

### `model`[^3]
- **model_name**: Specifies the name of the model, e.g., `"llama-3.1-8b"`.
- **model_path**: Path to the local model directory. Must be set for local models.
- **device**: Sets the computation device, e.g., `"cuda:0"`.
- **batch_size**: Batch size for training.
- **num_train_epochs**: Number of training epochs.
- **warmup_steps**: Number of warmup steps for training.
- **weight_decay**: Weight decay for regularization.
- **learning_rate**: Learning rate for optimization.
- **logging_steps**: Frequency of logging.
- **logging_dir**: Directory for saving training logs.
- **output_dir**: Directory for saving trained model outputs.
  
### `lora`[^4]
- **r**: LoRA rank, which determines the dimension of rank-reduced matrices.
- **lora_alpha**: Scaling factor for LoRA layers.
- **target_modules**: List of modules to which LoRA is applied.
- **lora_dropout**: Dropout rate for LoRA.
- **bias**: Bias handling in LoRA, either `"none"` or other options.

### `inference`[^1]
- **max_length**: Maximum sequence length for inference.
- **num_return_sequences**: Number of output sequences to return.
- **temperature**: Controls randomness in generated sequences.
- **top_p**: Cumulative probability for top-p sampling.
- **repetition_penalty**: Penalty for repeated tokens.
- **length_penalty**: Penalty for sequence length.
- **max_suffix_length**: Maximum suffix length during inference.

### `orpo-training`[^5]
- **beta**: ORPO training parameter, controlling trade-off between prompt and suffix fidelity.
- **num_train_epochs**: Number of training epochs for ORPO.
- **warmup_steps**: Warmup steps for ORPO training.
- **weight_decay**: Weight decay for ORPO training.
- **learning_rate**: Learning rate for ORPO training.
- **logging_steps**: ORPO training logging frequency.
- **logging_dir**: Directory for ORPO training logs.
- **output_dir**: Directory for ORPO trained model outputs.
- **batch_size**: Batch size for ORPO training.

### ORPO LoRA Configuration (`lora`)[^4]
- **r**: LoRA rank for ORPO model.
- **lora_alpha**: Scaling factor for ORPO LoRA layers.
- **target_modules**: Targeted modules for LoRA.
- **lora_dropout**: Dropout rate in LoRA for ORPO.
- **bias**: Handling for LoRA bias in ORPO, e.g., `"none"`.

---

[^1]: Information about these hyperparameters can be found here: https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/text_generation#transformers.GenerationConfig
[^2]: Information about these hyperparameters can be found here: https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
[^3]: Information about these hyperparameters can be found here https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/trainer#transformers.TrainingArguments
[^4]: Information about these hyperparameters can be found here: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
[^5]: Information about these hyperparameters can be found here: https://huggingface.co/docs/trl/main/en/orpo_trainer#trl.ORPOTrainer
