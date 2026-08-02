
# When Models Vanish: The Reproduction Window in LLM Behavior Research
## Overview

This repository contains the code and supporting materials for **When Models Vanish: The Reproduction Window in LLM Behavior Research**, an end-to-end reproduction of the ELEPHANT social-sycophancy benchmark. The project examines how reliably behavioral findings can be reproduced when the original language models, provider endpoints, and evaluation settings have changed or are no longer available.

We regenerate model responses across four behavioral datasets, rescore them from raw outputs using LLM evaluators, and compare the resulting measurements with the original study. The repository also includes robustness analyses covering evaluator identity, prompt versions, sampling temperature, maximum-token limits, and quantization.

Because several original models were unavailable at the time of reproduction, the experiments use a combination of direct model access, successor models, and approximate replacements accessed through multiple providers. The model identifiers, providers, scripts, and generation parameters used in each experiment are documented below.

## Repository Structure

```text
code/           Response-generation, response-evaluation scripts, and results analysis
data/           Data used for generation and evaluation
experiments/    Sensitivity, consistency, and robustness experiments
results/        Full model responses, scores, metadata, and result tables
full_analysis/  Main, appendix, and judge-robustness analysis notebooks
figures/        Generated figures and tables
```

Each experiment under [`experiments`](experiments) contains its own code, sample data, results, analysis scripts, and README when applicable.

## High-Level Workflow

1. **Generate model responses** using the scripts in [`code/response_generation`](code/response_generation).
2. **Score the responses with a judge** using the scripts in [`code/response_evaluation`](code/response_evaluation).
3. **Assemble the full result tables** by combining the model responses, judge scores, and associated metadata into the tables stored in [`results`](results).
4. **Run the results analysis notebooks** in [`code/results_analysis`](code/results_analysis) to reproduce the analyses, figures, and tables.

## Tracked Metadata

The generation and evaluation scripts preserve metadata alongside each response and score to support auditing and reproducibility. Available fields vary slightly by provider and inference backend, but generally include:

- **Data provenance:** input dataset path, prompt and response columns, row or request identifier, prompt text, and prompt hash.
- **Model identity:** requested model, returned model snapshot or revision, provider, quantization, model family, and inference backend when available.
- **Generation configuration:** temperature, top-p, seed, maximum tokens, reasoning configuration, number of requested completions, and whether the AITA binary instruction was used.
- **Execution provenance:** request, batch, and run IDs; request and API-call timestamps; Git commit; package versions; and provider-specific system fingerprints.
- **Token usage:** input or prompt tokens, output or completion tokens, total tokens, and reasoning-token usage when available.
- **Completion state:** finish reason, status, error message, retry or rerun count, and provider-specific generation metadata.
- **Local inference environment:** GPU name and count, numerical precision, model revision, and relevant Transformers, PyTorch, and Hugging Face Hub versions when applicable.
- **Judge audit fields:** judge model and provider, judge prompt, evaluated response, raw judge output, parsed binary score, judge sampling settings, token usage, timestamps, status, and errors.

Provider-specific scripts may record additional metadata, such as OpenRouter generation IDs, response IDs, artifact paths, provider-routing information, or estimated judge cost.

## Model and Evaluator Configurations

| Model | Platform | Type | Model ID | Script | Temperature | Top-p | Seed | Max tokens | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|
| GPT-5 | OpenAI | Direct | `gpt-5-2025-08-07` | `code/response_generation/openAI_response_extraction.py` | Default | Default | 123 | 5,000/6,000 | Initial limit was 5,000 tokens. Token-limit blanks were rerun with 6,000 tokens. Default reasoning effort was used. |
| GPT-4o | OpenAI | Direct | `gpt-4o-2024-11-20` | `code/response_generation/openAI_response_extraction.py` | Default | Default | 123 | 512 | Pinned OpenAI snapshot. |
| Gemini 3.5 Flash | Google | Successor | `gemini-3.5-flash` | `code/response_generation/gemini_response_batch_extraction.py` | Default | Default | 123 | 5,000 | Replacement for Gemini 1.5 Flash. Default reasoning effort was used. |
| Claude Sonnet 4.6 | Anthropic | Successor | `claude-sonnet-4-6` | `code/response_generation/claude_response_extraction.py` | Default | Default | — | 256 | Replacement for Claude Sonnet 3.7. |
| Llama-3-8B-Instruct | Hugging Face | Approximate | `NousResearch/Meta-Llama-3-8B-Instruct` | `code/response_generation/llama_hf_extraction_v2.py` | 0.6 | 0.9 | 123 | 500 | Loaded from a Hugging Face mirror repository. The model and sampling settings are defined in the script. |
| Llama-4-Scout-17B-16E | OpenRouter | Approximate | `meta-llama/llama-4-scout-17b-16e-instruct` | `code/response_generation/openrouter_response_extraction.py` | 0.6 | 0.9 | 123 | 512 | Underlying provider: DeepInfra. |
| Llama-3.3-70B-Instruct-Turbo | Together AI | Direct | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `code/response_generation/together_API_response_extraction.py` | 0.6 | 0.9 | 123 | 512 | Accessed directly through Together AI. |
| Mistral-7B-Instruct-v0.3 | Hugging Face | Direct | `mistralai/Mistral-7B-Instruct-v0.3` | `code/response_generation/extract_mistral_hf_v2.py` | 0.6 | 0.9 | 123 | 500 | The model and sampling settings are defined in the script. |
| Mistral-Small-24B-Instruct-2501 | OpenRouter | Approximate | `mistralai/mistral-small-24b-instruct-2501` | `code/response_generation/openrouter_response_extraction.py` | 0.6 | 0.9 | 123 | 512 | Underlying provider: DeepInfra. |
| DeepSeek-V3 | OpenRouter | Approximate | `deepseek/deepseek-chat-v3` | `code/response_generation/openrouter_response_extraction.py` | 0.6 | 0.9 | 123 | 512 | Underlying provider: DeepInfra. |
| Qwen2.5-7B-Instruct-Turbo | Together AI | Direct | `Qwen/Qwen2.5-7B-Instruct-Turbo` | `code/response_generation/together_API_response_extraction.py` | 0.6 | 0.9 | 123 | 512 | Accessed directly through Together AI. |
| GPT-4o judge | OpenAI | Evaluator | `gpt-4o-2024-11-20` | `code/response_evaluation/judge_scorer_extraction_GPT4o.py` | 0 | — | 123 | 2 | Binary judge output. |
| Llama-70B judge | OpenRouter | Evaluator | `meta-llama/llama-3.3-70b-instruct` | `code/response_evaluation/judge_scorer_extraction_llama70b.py` | 0 | — | 123 | 2 | Underlying provider: DeepInfra. Binary judge output. |

## API Credentials

Set the environment variables required by the providers you intend to use:

```bash
export OPENAI_API_KEY=<openai_api_key>
export ANTHROPIC_API_KEY=<anthropic_api_key>
export GEMINI_API_KEY=<gemini_api_key>
export OPENROUTER_API_KEY=<openrouter_api_key>
export TOGETHER_API_KEY=<together_api_key>
```

`GOOGLE_API_KEY` may be used instead of `GEMINI_API_KEY` for Gemini. The scripts also support the following local key-file fallbacks when the corresponding environment variable is not set:

| Provider | Environment variable | Local key-file fallback |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `key.txt` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic_key.txt` or `claude_key.txt` |
| Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini_key.txt` or `google_key.txt` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter_key.txt` or `openrouter.txt` |
| Together AI | `TOGETHER_API_KEY` | `together_key.txt` |

Environment variables are recommended. Never commit API keys or local key files to version control. Ollama and the local Hugging Face generation scripts do not use these provider API keys.

## Quick Start

The following example generates GPT-4o responses for a sample dataset and then scores those responses with the GPT-4o judge. Run the commands from the repository root.

First, set your OpenAI API key:

```bash
export OPENAI_API_KEY=<openai_api_key>
```

Generate responses using the GPT-4o configuration listed above. Temperature and top-p are omitted so the OpenAI API defaults are used.

```bash
python code/response_generation/openAI_response_extraction.py \
  --model_name gpt-4o-2024-11-20 \
  --input_file sample_datasets/AITA-YTA_sample.csv \
  --prompt_column prompt \
  --output_file quickstart/gpt4o_responses.csv \
  --output_column GPT-4o \
  --max_tokens 512 \
  --seed 123
```

Score the generated responses with the GPT-4o judge using temperature 0 and seed 123:

```bash
python code/response_evaluation/judge_scorer_extraction_GPT4o.py \
  --input_file quickstart/gpt4o_responses.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT4o_quickstart_scored \
  --output_file quickstart/gpt4o_responses_scored.csv \
  --temperature 0 \
  --seed 123
```

Both commands use the OpenAI Batch API. They print a batch ID that can be supplied with `--batch_id <batch_id>` to resume polling if a run is interrupted.

## API Costs

Model-response extraction costs vary by provider, model, dataset size, output length, and current API pricing. GPT-4o judge scoring for the complete set of responses from all 11 models cost approximately **$550**.

## Code Usage

Run the following commands from the repository root. Replace values enclosed in angle brackets with the appropriate model names, file paths, column names, and configuration values.

### Response Generation

#### OpenAI

```bash
python code/response_generation/openAI_response_extraction.py \
  --model_name <model_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```

#### Anthropic Claude

```bash
python code/response_generation/claude_response_extraction.py \
  --model_name <model_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p>
```

#### Google Gemini

```bash
python code/response_generation/gemini_response_batch_extraction.py \
  --model_name <model_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```

#### Ollama

```bash
python code/response_generation/llama_response_extraction.py \
  --model_name <model_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```
Note: This response generation was not used for the main reproduction

#### Hugging Face Llama

```bash
python code/response_generation/llama_hf_extraction_v2.py \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column>
```

#### Hugging Face Mistral

```bash
python code/response_generation/extract_mistral_hf_v2.py \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --checkpoint_every <checkpoint_interval> \
  --max_retries <maximum_retries>
```

The Hugging Face generation scripts use the model and sampling settings defined in each script. They require a compatible local GPU environment. For the Mistral extractor, use `--row_limit <number_of_rows>` for a partial run or `--resume` to continue from its saved CSV and RNG checkpoint.

#### OpenRouter

```bash
python code/response_generation/openrouter_response_extraction.py \
  --model_name <model_name> \
  --provider_name <provider_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```

#### Together AI

```bash
python code/response_generation/together_API_response_extraction.py \
  --model_name <model_name> \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --output_file <output_csv> \
  --output_column <response_column> \
  --max_tokens <maximum_output_tokens> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```

Generation arguments such as `--temperature`, `--top_p`, `--seed`, and `--output_column` are optional. Omit an optional sampling argument to use the provider's default behavior. Add `--AITA_binary` when binary AITA classification is required. Batch-based scripts print a batch ID that can be supplied with `--batch_id <batch_id>` to resume an interrupted run.

### Response Evaluation

#### GPT-4o Judge

```bash
python code/response_evaluation/judge_scorer_extraction_GPT4o.py \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --response_column <response_column> \
  --output_column_tag <output_column_tag> \
  --output_file <output_csv> \
  --temperature <temperature> \
  --seed <seed>
```

#### Llama-70B Judge

```bash
python code/response_evaluation/judge_scorer_extraction_llama70b.py \
  --input_file <input_csv> \
  --prompt_column <prompt_column> \
  --response_column <response_column> \
  --output_column_tag <output_column_tag> \
  --output_file <output_csv> \
  --model_name <judge_model_name> \
  --provider_name <provider_name> \
  --temperature <temperature> \
  --top_p <top_p> \
  --seed <seed>
```

Both evaluation scripts score validation, indirectness, and framing when no individual metric flag is supplied. Use `--validation`, `--indirectness`, or `--framing` to run only selected metrics. Optional sampling arguments may be omitted to use the provider's default behavior.

## Full Analysis

The complete analysis is available in the [`full_analysis`](full_analysis) directory.

- [`paper_reproduction_analysis.ipynb`](full_analysis/paper_reproduction_analysis.ipynb) contains the main reproduction analysis.
- [`additional_graphs_appendix.ipynb`](full_analysis/additional_graphs_appendix.ipynb) contains the supplementary and appendix analyses and generates the additional figures and tables.
- [`Judge Robustness`](full_analysis/Judge%20Robustness) contains the analyses comparing GPT-4o and Llama-70B judge results:
  - [`reproduction_robustness_analysis.ipynb`](full_analysis/Judge%20Robustness/reproduction_robustness_analysis.ipynb) analyzes how reproduction results change across judges.
  - [`judge_robustness_agreement.ipynb`](full_analysis/Judge%20Robustness/judge_robustness_agreement.ipynb) evaluates judge agreement, precision, and recall.
- [`Appendix Info`](full_analysis/Appendix%20Info) contains supporting summary CSVs used by the appendix analysis.

Generated figures are organized by analysis type in the [`figures`](figures) directory.

## Results

The [`results`](results) directory contains the complete result tables used by the analysis notebooks and figures. The result CSVs include the full model responses alongside their scores and associated metadata. Results are grouped by their role in the study:

- [`paper_full_results`](results/paper_full_results) contains the metric tables released by the original ELEPHANT study. These files serve as the reference results for the reproduction comparisons.
- [`main_reproduction_full_results`](results/main_reproduction_full_results) contains the complete results from the main reproduction, including dataset-level scores, the AITA-NTA moral-sycophancy calculations, and GPT-4o-rescored human baselines in its [`human_baseline`](results/main_reproduction_full_results/human_baseline) subfolder.
- [`judge_robustness_full_results`](results/judge_robustness_full_results) contains results rescored with the alternate Llama-70B judge. Its [`human_baseline`](results/judge_robustness_full_results/human_baseline) subfolder contains the corresponding alternate-judge human-baseline scores.

Files are organized by dataset, including AITA-YTA, AITA-NTA-OG, AITA-NTA-FLIP, OEQ, and SS. The analysis notebooks read these tables to compare the original study, the main reproduction, and the judge-robustness evaluation.

**Dataset naming note:** SS and ALP refer to the same dataset. The original paper used both names; consistent with that convention, we refer to the dataset as **ALP** in our paper and as **SS** throughout this repository.

## Citation of the Reproduced Work

Myra Cheng, Sunny Yu, Cinoo Lee, Pranav Khadpe, Lujain Ibrahim, and Dan Jurafsky. “ELEPHANT: Measuring and Understanding Social Sycophancy in LLMs.” In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2026b. [arXiv:2505.13995](https://arxiv.org/abs/2505.13995).
