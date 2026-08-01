# Quantization Sensitivity

## Overview

This experiment evaluates whether model quantization affects measured sycophancy behavior for **Llama-3-8B**.

The experiment was motivated by the availability of multiple quantized versions of Llama-3-8B through Ollama before obtaining the model mirror used in the main reproduction. Because quantization changes how model weights are represented, it may alter generated responses even when the model family, prompts, and generation settings remain fixed.

To test this, responses were generated under two quantization conditions:

1. **Q4 quantization**
2. **Q8 quantization**

Each quantization was evaluated under two sampling configurations:

1. **Temperature 0**
2. **Paper sampling parameters:** temperature `0.6` and top-p `0.9`

The same sampled examples, prompts, seed, token limit, and scoring procedure were used across all four conditions.

## Research Question

**How sensitive are Llama-3-8B sycophancy scores to model quantization and sampling configuration?**

This experiment also examines whether either quantization more closely reproduces the scores released by the original study.

## Experimental Design

A random sample of examples from the **AITA-YTA** dataset was evaluated under four generation conditions:

1. **Q4 at temperature 0**
2. **Q8 at temperature 0**
3. **Q4 using the paper sampling parameters**
4. **Q8 using the paper sampling parameters**

The following settings were held constant across conditions:

- Dataset examples
- Prompt
- Model family
- Random seed
- Maximum-token limit
- Scoring prompts
- Judge model and configuration

The generated responses were scored for:

- **Validation**
- **Indirectness**
- **Framing**

Because the same examples were used across all four conditions, comparisons are treated as paired analyses.

## Experimental Conditions

### Condition 1: Q4 at Temperature 0

Responses were generated using the Q4 quantization.

### Condition 2: Q8 at Temperature 0

The same prompts were generated using the Q8 quantization.

### Condition 3: Q4 With Paper Sampling Parameters

Responses were generated using the Q4 quantization with:

- **Temperature:** `0.6`
- **Top-p:** `0.9`

### Condition 4: Q8 With Paper Sampling Parameters

Responses were generated using the Q8 quantization with:

- **Temperature:** `0.6`
- **Top-p:** `0.9`

## Configuration

### Data Configuration

- **Dataset:** `AITA-YTA`
- **Sample size:** `200`
- **Sampling method:** `random sample`
- **Prompt source:** `sample/AITA-YTA_sample_200_quantization_experiment.csv`
- **Response model:** `Llama-3-8B`

### Model Configuration

- **Model family:** `Llama-3-8B`
- **Model distribution:** `Ollama`
- **Quantizations:**
  - `Q4`
  - `Q8`
- **Model identifiers Q4:** `llama3:8b-instruct-q4_0`
- **Model identifiers Q8:** `llama3:8b-instruct-q8_0`
- **Maximum output tokens:** `500`
- **Seed:** `123`

### Sampling Configurations

#### Temperature-0 Configuration

- **Temperature:** `0`
- **Top-p:** `default`

#### Paper-Parameter Configuration

- **Temperature:** `0.6`
- **Top-p:** `0.9`

### Judge Configuration

- **Judge model:** `GPT-4o`
- **Judge snapshot:** `gpt-4o-2024-11-20`
- **API provider:** `OpenAI`
- **Temperature:** `0`
- **Seed:** `123`
- **Maximum output tokens:** `2`
- **Dimensions evaluated:**
  - Validation
  - Indirectness
  - Framing
- **Output format:** `csv`

## Experimental Setup

### Run Q4 at Temperature 0

```bash
python code/llama_quantization_response_extraction.py \
  --test_data sample/AITA-YTA_sample_200_quantization_experiment.csv \
  --prompt_column prompt \
  --model_name llama3:8b-instruct-q4_0 \
  --output_file <path_to_q4_temp0_responses.csv> \
  --temperature 0 \
  --seed 123 \
  --max_tokens 500
```

### Run Q8 at Temperature 0

```bash
python code/llama_quantization_response_extraction.py \
  --test_data sample/AITA-YTA_sample_200_quantization_experiment.csv \
  --prompt_column prompt \
  --model_name llama3:8b-instruct-q8_0 \
  --output_file <path_to_q8_temp0_responses.csv> \
  --temperature 0 \
  --seed 123 \
  --max_tokens 500
```

### Run Q4 With Paper Parameters

```bash
python code/llama_quantization_response_extraction.py \
  --input_file sample/AITA-YTA_sample_200_quantization_experiment.csv \
  --prompt_column prompt \
  --model llama3:8b-instruct-q4_0 \
  --output_file <path_to_q4_paper_responses.csv> \
  --temperature 0.6 \
  --top_p 0.9 \
  --seed 123 \
  --max_tokens 500
```

### Run Q8 With Paper Parameters

```bash
python code/llama_quantization_response_extraction.py \
  --input_file sample/AITA-YTA_sample_200_quantization_experiment.csv \
  --prompt_column prompt \
  --model llama3:8b-instruct-q8_0 \
  --output_file <path_to_q8_paper_responses.csv> \
  --temperature 0.6 \
  --top_p 0.9 \
  --seed 123 \
  --max_tokens 500
```

## Scoring

Each generated response file is scored for validation, indirectness, and framing using the same judge configuration.

### Scoring All Responses

```bash
python code/llama_quantization_scorer.py \
  --input_file <path_to_responses.csv> \
  --prompt_column prompt \
  --response_column <response_column> \
  --output_file <path_to_scored.csv> \
  --temperature 0 \
  --seed 123
```

## Analysis

The analysis includes four primary paired comparisons:

1. **Q4 vs. Q8 at temperature 0**
2. **Q4 vs. Q8 under the paper sampling parameters**
3. **Q4 at temperature 0 vs. Q4 under the paper parameters**
4. **Q8 at temperature 0 vs. Q8 under the paper parameters**

Each of the four experimental conditions is also compared with the corresponding scores released by the original study.

For each sycophancy dimension, the analysis compares:

- Mean score under each condition
- Mean difference between conditions
- Direction of label changes
- Paired agreement rate
- Statistical significance of directional differences
- Practical equivalence within the prespecified margin

The following dimensions are analyzed separately:

- **Validation**
- **Indirectness**
- **Framing**

## Statistical Tests

### Directional Differences

Exact McNemar tests are used to determine whether discordant binary labels change systematically in one direction between paired conditions.

A directional difference is treated as statistically significant when:

```text
p < 0.05
```

### Practical Equivalence

Two one-sided tests are used to evaluate whether paired score differences fall within the prespecified practical-equivalence margin.

The equivalence margin is:

```text
±0.05
```

Practical equivalence is established when the TOST procedure is significant at:

```text
α = 0.05
```

Statistical nonsignificance under McNemar’s test does not by itself establish practical equivalence.

## Run the Analysis

Comparison between Q4/Q8 variations:

```bash
python analysis/compare_scored_outputs.py \
  --pair q4_vs_q8_temp0 \
    results/llama3_q4_scored.csv \
    results/llama3_q8_scored.csv \
  --pair q4_vs_q8_paper \
    results/llama3_q4_paper_parameters_scored.csv \
    results/llama3_q8_paper_parameters_scored.csv \
  --pair q4_temp0_vs_paper \
    results/llama3_q4_scored.csv \
    results/llama3_q4_paper_parameters_scored.csv \
  --pair q8_temp0_vs_paper \
    results/llama3_q8_scored.csv \
    results/llama3_q8_paper_parameters_scored.csv \
  --output-file analysis/llama_quantization_pairwise_comparisons.csv
```

Comparison aganist original scores:
```bash
python analysis/compare_scored_outputs.py \
  --pair q4_temp0_vs_original_gpt4o \
    results/llama3_q4_scored.csv \
    sample/AITA-YTA_sample_200_with_full_results_quantization_experiment.csv \
  --pair q8_temp0_vs_original_gpt4o \
    results/llama3_q8_scored.csv \
    sample/AITA-YTA_sample_200_with_full_results_quantization_experiment.csv \
  --pair q4_paper_vs_original_gpt4o \
    results/llama3_q4_paper_parameters_scored.csv \
    sample/AITA-YTA_sample_200_with_full_results_quantization_experiment.csv \
  --pair q8_paper_vs_original_gpt4o \
    results/llama3_q8_paper_parameters_scored.csv \
    sample/AITA-YTA_sample_200_with_full_results_quantization_experiment.csv \
  --file-b-score-template '{metric}_GPT-4o' \
  --output-file analysis/quantization_vs_original_gpt4o_comparisons.csv
  ```


## Analysis Outputs

The analysis produces outputs for:

- **Q4 vs. Q8 at temperature 0**
- **Q4 vs. Q8 under the paper parameters**
- **Q4 temperature 0 vs. Q4 paper parameters**
- **Q8 temperature 0 vs. Q8 paper parameters**
- **Each experimental condition vs. the original released scores**
- **Practical-equivalence tests**

## Outputs

- **Sampled prompts:** `sample/AITA-YTA_sample_200_quantization_experiment.csv`
- **Q4 temperature-0 responses:** `results/llama3_q4_scored.csv`
- **Q8 temperature-0 responses:** `results/llama3_q8_scored.csv`
- **Q4 paper-parameter responses:** `results/llama3_q4_paper_parameters_scored.csv`
- **Q8 paper-parameter responses:** `esults/llama3_q8_paper_parameters_scored.csv`
- **Pairwise comparison results:** `analysis/llama_quantization_pairwise_comparisons.csv`
- **Original-score comparisons:** `analysis/llama_quantization_vs_full_results_llama8b_comparisons.csv`
- **Analysis notebook:** `../../full_analysis/additional_graphs_appendix.ipynb`
- **Mean-score figure:** `../../figures/appendix_figures/llama_quantization_mean_scores.pdf`
- **Pairwise Comparison Table:** `../../figures/appendix_figures/llama_quantization_pairwise_comparisons_table.pdf`
- **Original-minus-experimental figure:** `../../figures/appendix_figures/llama_original_minus_experimental_mean_differences.pdf`

## Interpretation

Quantization may affect behavioral benchmark scores even when the model family, prompts, seed, token limit, and scoring configuration remain unchanged.

This experiment evaluates both aggregate score differences and response-level agreement because similar means can conceal substantial variation in the labels assigned to individual examples.


Validation and framing were comparatively stable across quantizations, while indirectness was consistently higher under Q8, showing that quantization can materially affect measured model behavior.
