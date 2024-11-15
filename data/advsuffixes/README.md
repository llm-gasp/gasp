# AdvSuffixes - Information

AdvSuffixes is a curated dataset of adversarial prompts and suffixes designed to evaluate and enhance the robustness of large language models (LLMs) against adversarial attacks. By appending these suffixes to standard prompts, researchers and developers can explore and analyze how LLMs respond to potentially harmful input scenarios. This dataset is heavily inspired by [AdvBench](https://github.com/llm-attacks/llm-attacks).

## Dataset Structure
The dataset is organized as follows:
```
data/
│
├── advsuffixes/
│   ├── advsuffixes.csv      # Adversarial suffixes and their respective prompts
│   ├── advsuffixes_eval.txt # 100 additional evaluation prompts that are out-of-distribution
│
├── ...
```

- `advsuffixes.csv`: The primary dataset containing pairs of adversarial suffixes and corresponding prompts.  
- `advsuffixes_eval.txt`: A set of 100 additional evaluation prompts designed to test model robustness, that are out-of-distribution from the original 519 prompts in `advsuffixes.csv`. 

Details about the dataset generation have been provided in Appendix B in the supplementary materials of the paper. There are 11763 listed suffixes overall, averaging 22.6 suffixes per prompt.

---

## License
This dataset is distributed under the GNU General Public License v3.0.