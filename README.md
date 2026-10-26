# Budget Force Reasoning

This repository contains a small GSM8K experiment comparing ordinary generation with budget forcing. The model is asked to continue reasoning after it first tries to stop, then its answers are scored under 0-wait and 2-wait settings.

The experiment uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and a QLoRA adapter trained on 450 examples. The held-out test set contains 90 examples across nine topics.

## Repository layout

- `base/` — notebook and inputs for evaluating the unmodified DeepSeek model. `results/` contains its saved metrics; `weights/` and `plots/` are created by the notebook.
- `budget_forcing/` — the training notebook (`budget_force_model.ipynb`), adapter-only evaluation notebook (`evaluation.ipynb`), final `train.json` and `test.json`, saved results, plots, and adapter/tokenizer files.
- `data/` — scripts used to prepare the data, prompt templates, source/intermediate files, and filtering utilities. `stage_1_split/` and `stage_2_split/` are intermediate data products.
- `data/filters/` — validation, de-duplication, length, difficulty, and balancing helpers used by the data scripts.

`train.json` and `test.json` beside each notebook are the final inputs for that notebook. They are deliberately duplicated there so they can be uploaded directly to Colab.

## Running the code

The notebooks are intended for Google Colab. Open the appropriate notebook, select **Runtime → Change runtime type → T4 GPU**, upload the files named in the section below, and run the cells from top to bottom. The notebooks install their own Python dependencies.

To rebuild the early data-preparation stage locally, create an environment with Python 3.10+ and install the required package:

```bash
pip install datasets
python data/build_stage_1.py --overwrite
```

This downloads GSM8K unless the source Parquet files are already in `data/raw/`. The resulting topic-label request files must be completed and validated before `data/build_stage_2.py` can produce the final balanced split. Most users can skip this process and use the checked-in final JSON files.

For a base-model evaluation, upload `base/base_model.ipynb` and `base/test.json` to Colab, then run every cell. Uploading `base/train.json` is optional; it is used only to show the training-set size. Results are written to `base/results/` and plots to `base/plots/`.

## Colab T4: adapter budget-forcing evaluation

There is no standalone `budget_force_evaluation.py` in this repository; the evaluation entry point is `budget_forcing/evaluation.ipynb`. Use it when an adapter has already been trained and you do not want to retrain it.

In a fresh Colab runtime with a T4 GPU, upload these files to `/content` (or preserve the same paths inside a `budget_forcing/` folder):

```text
evaluation.ipynb
test.json
weights/adapter_config.json
weights/adapter_model.safetensors
```

Upload the tokenizer files from the same training run as well when available:

```text
weights/tokenizer.json
weights/tokenizer_config.json
weights/special_tokens_map.json
weights/chat_template.jinja
```

The notebook downloads the base weights automatically from Hugging Face:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

It then loads those base weights in FP16, attaches the uploaded QLoRA adapter, and evaluates the uploaded `test.json` with 0 and 2 forced continuations. It writes `adapter_budget_results.json` and the corresponding plots under `results/` and `plots/`. A run typically takes about 15–25 minutes for 0 waits and 30–55 minutes for 2 waits on a T4. Do not upload the full base-model checkpoint: Colab downloads it itself. Keep the adapter configuration and adapter tensor from the same training run.

If you also upload `base_budget_results.json` from a base-model run, the notebook compares base and fine-tuned results only when both files use the same `test.json` hash.

To train a new adapter instead, upload `budget_forcing/budget_force_model.ipynb`, `budget_forcing/train.json`, and `budget_forcing/test.json`, then run all cells on a T4. It saves the adapter, tokenizer, metrics, and plots and offers a downloadable `budget_force_results.zip` archive.

## Notes on weights

Do not add full base-model snapshots or large adapter tensors to normal Git history. The project ignores downloaded checkpoints, and `budget_forcing/weights/adapter_model.safetensors` is configured for Git LFS. LFS storage and bandwidth limits still apply.

GSM8K is available under the MIT License. The model is [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), whose model card lists an MIT license. This is a small educational experiment, not a reproduction of the original s1 results.
