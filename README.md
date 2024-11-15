![code](./assets/code.png)

# GASP

This repo is an official implementation of GASP for the paper *GASP: Efficient Black-Box Generation of Adversarial Suffixes
for Jailbreaking LLMs* (currently in review at CVPR '25).

Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. 
Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.

![GASP](./assets/gasp.svg)

## 1. Installation
- Clone this repository to your local machine.
- Install all necessary dependencies on your machine. We would suggest using A100s for running GASP.

```bash
git clone https://github.com/llm-gasp/gasp
cd gasp
pip install -r requirements.txt
```

- To use a Docker image with all dependencies pre-installed, you can pull it directly from Docker Hub: `anonrsch/gasp:latest`. This setup is recommended when working with schedulers like SLURM.

### 1.1 Configuration Setup
Refer to the [config documentation](./config/README.md) for details on key configurations that can be modified when running GASP. The `./config/` folder contains four main configuration files. We also suggest [downloading](https://huggingface.co/docs/hub/en/models-downloading) the necessary LLMs from HuggingFace (for SuffixLLM, at least). You may reference these downloaded models in the configuration files.

GASP currently supports model loading and inference from two primary providers: OpenAI and HuggingFace, with plans to expand to additional providers such as Groq and Cerebras.

## 2. Execution
Three task options are supported: `all`, `train`, and `eval`. To run both `train` and `eval` together:

```bash
python3 gasp.py --task=all
```

To run only `train` or `eval`, use:

```bash
python3 gasp.py --task=train
python3 gasp.py --task=eval
```

## 3. License
Our source code is under the GNU General Public License v3.0.

## 4. Authors
Authors and contributors would be added here after acceptance.

## 5. Ethical Statement
Our research and the development of GASP are driven by the commitment to advancing the understanding of LLM vulnerabilities. While GASP enables the efficient generation of coherent adversarial suffixes, it is worth noting that manual methods for jailbreaking LLMs have already been widely accessible. Our research seeks to formalize and characterize these vulnerabilities rather than introduce novel threats. 

In alignment with responsible disclosure practices, we have shared our findings with relevant organizations whose models were tested in this study and transparently disclosed all of our findings. 


