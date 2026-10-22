# Budget Force Reasoning

Colab-ready experiments for a small GSM8K reasoning dataset: base-model inference, QLoRA fine-tuning, and s1-style budget forcing.

## Package contents

- `base/`: baseline notebooks and the final `train.json` / `test.json` splits.
- `budget_forcing/`: QLoRA training/evaluation notebook, adapter-only evaluation notebook, current run metrics/plots, and the same final splits.
- `data/`: the data-preparation scripts, filters, prompts, raw source files, and intermediate splits. The final `train/` and `test/` directories are omitted because those files are beside the model notebooks.

The final split currently contains 450 training examples and 90 test examples, balanced over nine topics. The test set has ten examples per topic.

## Run in Colab

For the controlled base evaluation, upload `base/base_model.ipynb` and `base/test.json` to Colab and run all cells on a T4 GPU (CPU fallback is supported but slow). `train.json` is optional and only used to display its size. The notebook evaluates both 0- and 2-Wait settings and saves JSON metrics under `base/results/` and plots under `base/plots/`. The public base model is downloaded into `base/weights/base_model_weights/` when it is not cached.

For training and budget-forcing evaluation, upload `budget_forcing/budget_force_model.ipynb`, `train.json`, and `test.json` to `/content`, then run all cells on a T4 GPU. It trains the QLoRA adapter, evaluates 0 and 2 forced continuations, saves plots under `budget_forcing/plots/`, metrics under `budget_forcing/results/`, and adapter/tokenizer files under `budget_forcing/weights/`. It downloads an archive with those same three subfolders; extract it into the local `budget_forcing/` folder to preserve the layout.

For evaluation without retraining, upload `budget_forcing/evaluation.ipynb`, `test.json`, and the adapter files `weights/adapter_config.json` and `weights/adapter_model.safetensors` from one training run (tokenizer files from `weights/` are also useful). Optionally upload `base/results/base_budget_results.json`; the notebook checks its test-file hash and prints a matched comparison. Run this notebook in a fresh Colab runtime to avoid keeping two model copies in T4 memory. It saves metrics under `results/` and plots under `plots/`. The base model downloads from Hugging Face if not cached.

Both evaluation notebooks use the same prompt, FP16 base weights, batch size, answer parsing, 1,024-token reasoning cap, 32-token answer allowance, and generation/scoring code. They record the test-file SHA-256 so mismatched splits are not compared. The package layout under `/content/base/` and `/content/budget_forcing/` is recognized; outputs are grouped into `plots/`, `weights/`, and `results/` beside the package files.

## Weights and Git

Do not commit model snapshots or adapter tensors to ordinary Git history. The base snapshot is several gigabytes, and the trained adapter file from prior runs is about 148 MB; GitHub blocks individual regular Git files above 100 MiB. The `.gitignore` excludes downloaded model snapshots and checkpoints, while `.gitattributes` tracks `budget_forcing/weights/adapter_model.safetensors` using Git LFS. Git LFS storage and bandwidth quotas apply. The training notebook downloads the base snapshot and exports the adapter separately for Colab.

The adapter files are in `budget_forcing/weights/`, metrics JSONs in `budget_forcing/results/`, and plots in `budget_forcing/plots/`. The evaluation-only notebook can load this adapter after you upload `test.json` and the adapter configuration/tensor to Colab. The training notebook exports this layout in its result archive.

## Data and model attribution

The raw GSM8K source dataset is distributed under the MIT License; see the [Hugging Face dataset card](https://huggingface.co/datasets/openai/gsm8k). The model referenced by the notebooks is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), listed under the MIT License by DeepSeek. Its model card also notes that the distilled Qwen base originates from Qwen2.5, originally licensed under Apache 2.0. Review upstream notices before redistributing model artifacts.

The notebook settings, dataset size, model size, and Colab hardware differ substantially from the original s1 experiments. Results should be presented as a small educational experiment, not a reproduction of s1's reported performance.
