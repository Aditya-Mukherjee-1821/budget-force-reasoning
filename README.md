# Budget Force Reasoning

Colab-ready experiments for a small GSM8K reasoning dataset: base-model inference, QLoRA fine-tuning, and s1-style budget forcing.

## Package contents

- `base/`: baseline notebooks and the final `train.json` / `test.json` splits.
- `budget_forcing/`: QLoRA training/evaluation notebook, adapter-only evaluation notebook, current run metrics/plots, and the same final splits.
- `data/`: the data-preparation scripts, filters, prompts, raw source files, and intermediate splits. The final `train/` and `test/` directories are omitted because those files are beside the model notebooks.

The final split currently contains 450 training examples and 90 test examples, balanced over nine topics. The test set has ten examples per topic.

## Run in Colab

For base inference, upload `base/base_model.ipynb` and `base/test.json` to Colab and run the notebook. `train.json` is optional and used only to print the train-set size. It downloads `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` into `/content/base_model_weights` when absent. Predictions, metrics, and plots are written directly under `/content`.

For training and budget-forcing evaluation, upload `budget_forcing/budget_force_model.ipynb`, `train.json`, and `test.json` to `/content`, then run all cells on a T4 GPU. It trains the QLoRA adapter, evaluates 0 and 2 forced continuations, saves plots and JSON metrics under `/content`, and downloads an archive containing the adapter, plots, and metrics. Extract the archive directly into the local `budget_forcing/` folder to place the adapter files beside the notebooks.

For evaluation without retraining, upload `budget_forcing/evaluation.ipynb`, `test.json`, `adapter_config.json`, and `adapter_model.safetensors` to `/content`. The adapter files can alternatively be uploaded together in an `sft_qlora/` folder. The notebook downloads the public base model if it is not cached and evaluates the test set at 0 and 2 forced continuations. Uploading only the test file and evaluation notebook is not enough for the fine-tuned result: the adapter contains the learned SFT changes. The base model alone can still be evaluated with the base notebook.

The notebooks also recognize the package layout under `/content/base/` and `/content/budget_forcing/`. When running them from `/content`, outputs are written directly under `/content` so they can be downloaded from the Colab file browser.

## Weights and Git

Do not commit model snapshots or adapter tensors to ordinary Git history. The base snapshot is several gigabytes, and the trained adapter file from prior runs is about 148 MB; GitHub blocks individual regular Git files above 100 MiB. The `.gitignore` excludes downloaded model snapshots and checkpoints, while `.gitattributes` tracks `budget_forcing/adapter_model.safetensors` using Git LFS. Git LFS storage and bandwidth quotas apply. The training notebook downloads the base snapshot and exports the adapter separately for Colab.

The fresh run's adapter files, tokenizer files, JSON metrics, and plots are now included in `budget_forcing/`. The evaluation-only notebook can load this adapter after you upload `test.json`, `adapter_config.json`, and `adapter_model.safetensors` to Colab. The updated training notebook also exports these files in its result archive.

## Data and model attribution

The raw GSM8K source dataset is distributed under the MIT License; see the [Hugging Face dataset card](https://huggingface.co/datasets/openai/gsm8k). The model referenced by the notebooks is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), listed under the MIT License by DeepSeek. Its model card also notes that the distilled Qwen base originates from Qwen2.5, originally licensed under Apache 2.0. Review upstream notices before redistributing model artifacts.

The notebook settings, dataset size, model size, and Colab hardware differ substantially from the original s1 experiments. Results should be presented as a small educational experiment, not a reproduction of s1's reported performance.
