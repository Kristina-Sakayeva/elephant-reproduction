
# When Models Vanish: The Reproduction Window in LLM Behavior Research
## Overview

This repository contains the code and supporting materials for **When Models Vanish: The Reproduction Window in LLM Behavior Research**, an end-to-end reproduction of the ELEPHANT social-sycophancy benchmark. The project examines how reliably behavioral findings can be reproduced when the original language models, provider endpoints, and evaluation settings have changed or are no longer available.

We regenerate model responses across four behavioral datasets, rescore them from raw outputs using LLM evaluators, and compare the resulting measurements with the original study. The repository also includes robustness analyses covering evaluator identity, prompt versions, sampling temperature, maximum-token limits, and quantization.

Because several original models were unavailable at the time of reproduction, the experiments use a combination of direct model access, successor models, and approximate replacements accessed through multiple providers. The model identifiers, providers, scripts, and generation parameters used in each experiment are documented below.

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
