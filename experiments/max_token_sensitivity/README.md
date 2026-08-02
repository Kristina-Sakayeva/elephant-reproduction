# Maximum-Token Sensitivity

## Overview

This experiment evaluates whether the maximum number of tokens allowed during response generation affects the resulting sycophancy scores.

The analysis was motivated by a discrepancy between the generation limit specified in the released code and the lengths of some responses included in the released results. For GPT-4o on the **AITA-NTA-FLIP** dataset, the code specified a maximum of 512 tokens, while some released responses appeared to exceed that limit.

To test whether the generation limit materially affected the outputs, the same sample of prompts was evaluated under two maximum-token conditions:

1. **512-Token Condition:** Responses were generated using the limit specified in the released code.
2. **2,000-Token Condition:** Responses were generated using a larger limit that exceeded the length of the longest released response.

All other generation settings were held constant.

## Research Question

**How sensitive are model responses and resulting sycophancy scores to the maximum-token generation limit?**

This experiment also examines whether either token-limit condition produces scores that align more closely with the scores released by the original study.

## Experimental Design

A random sample of examples from the **AITA-NTA-FLIP** dataset was selected.

For each sampled prompt, GPT-4o generated one response under each of the following conditions:

- **512 maximum tokens**
- **2,000 maximum tokens**

The following settings were held constant across both conditions:

- Prompt
- Sampled examples
- Model snapshot
- Temperature
- Random seed

The generated responses were then scored for:

- **Validation**
- **Indirectness**
- **Framing**

Because the same prompts were used in both conditions, the comparison between token limits is treated as a paired analysis.

## Experimental Conditions

### Condition 1: 512-Token Limit

Responses were generated using a maximum output length of 512 tokens, matching the value specified in the released code.

### Condition 2: 2,000-Token Limit

The same prompts were used to generate responses with a maximum output length of 2,000 tokens.

This limit was selected to exceed the length of the longest GPT-4o response observed in the released results.

## Configuration

### Data Configuration

- **Dataset:** `AITA-NTA-FLIP`
- **Sample size:** `200`
- **Sampling method:** `random sample`
- **Prompt source:** `sample/AITA-NTA-FLIP_sample.csv`
- **Response model:** `GPT-4o`

### Generation Configuration

- **Model:** `GPT-4o`
- **Model snapshot:** `gpt-4o-2024-11-20`
- **API provider:** `OpenAI`
- **Temperature:** `0`
- **Seed:** `123`
- **Maximum-token conditions:**
  - `512`
  - `2000`
- **Batching method:** `batch API`

### Judge Configuration

- **Judge model:** `GPT-4o`
- **Judge snapshot:** `gpt-4o-2024-11-20`
- **API provider:** `OpenAI`
- **Temperature:** `0`
- **Maximum output tokens:** `2`
- **Dimensions evaluated:**
  - Validation
  - Indirectness
  - Framing
- **Output format:** `csv`

## Experimental Setup

### First Run: 512-Token Condition

```bash
python code/max_token_count_GPT_extraction.py \
  --input_file sample/AITA-NTA-FLIP_sample.csv \
  --prompt_column flipped_story \
  --output_file results/AITA-NTA-FLIP_gpt4o_temp0_512_token_responses_seed_123.csv \
  --model_name gpt-4o-2024-11-20 \
  --temperature 0 \
  --seed 123 \
  --max_tokens 512 \
  --output_column GPT4o_temp0_AITA-NTA-FLIP_response_512_token
```

### Second Run: 2000-Token Condition

```bash
python code/max_token_count_GPT_extraction.py \
  --input_file sample/AITA-NTA-FLIP_sample.csv \
  --prompt_column flipped_story \
  --output_file results/AITA-NTA-FLIP_gpt4o_temp0_2000_token_responses_seed_123.csv \
  --model_name gpt-4o-2024-11-20 \
  --temperature 0 \
  --seed 123 \
  --max_tokens 2000 \
  --output_column GPT4o_temp0_AITA-NTA-FLIP_response_2000_token
```
### Score the 512-Token Run

```bash
python code/max_token_count_scorer.py \
  --input_file results/AITA-NTA-FLIP_gpt4o_temp0_512_token_responses_seed_123.csv \
  --prompt_column flipped_story \
  --response_column GPT4o_temp0_AITA-NTA-FLIP_response_512_token \
  --output_column_tag GPT4o_AITA-NTA-FLIP_temp0_512_token_seed_123 \
  --output_file results/AITA-NTA-FLIP_gpt4o_temp0_512_token_scored_seed_123.csv \
  --temperature 0
```

### Score the 2,000-Token Run

```bash
python code/max_token_count_scorer.py \
  --input_file results/AITA-NTA-FLIP_gpt4o_temp0_2000_token_responses_seed_123.csv \
  --prompt_column flipped_story \
  --response_column GPT4o_temp0_AITA-NTA-FLIP_response_2000_token \
  --output_column_tag GPT4o_AITA-NTA-FLIP_temp0_2000_token_seed_123 \
  --output_file results/AITA-NTA-FLIP_gpt4o_temp0_2000_token_scored_seed_123.csv \
  --temperature 0
```

## Analysis

The analysis includes three paired comparisons:

1. **512-token condition vs. 2,000-token condition**
2. **512-token condition vs. the original released scores**
3. **2,000-token condition vs. the original released scores**

### Token-Limit Comparison

The direct comparison between the 512-token and 2,000-token conditions evaluates whether changing the maximum generation length affects the resulting sycophancy scores.

For each dimension, the analysis calculates:

- Mean score under the 512-token condition
- Mean score under the 2,000-token condition
- Mean difference between conditions
- Number of changes from negative to positive
- Number of changes from positive to negative
- Exact paired-test p-value

The following dimensions are analyzed separately:

- **Validation**
- **Indirectness**
- **Framing**

Because the same prompts were used in both conditions, the analysis treats the scores as paired observations. The direction of each disagreement is recorded to determine whether increasing the token limit produces a systematic increase or decrease in a given score.

### Comparison With the Original Scores

Each token-limit condition is also compared with the corresponding scores released by the original study.

For both the 512-token and 2,000-token conditions, the analysis calculates:

- Agreement rate for each dimension
- Overall agreement across dimensions
- Number and direction of label changes
- Mean reproduced score
- Mean original score

These comparisons evaluate whether either token-limit condition more closely reproduces the original released evaluation.

### Paired Agreement Comparison

A separate paired comparison evaluates whether the 512-token condition agrees with the original scores more often than the 2,000-token condition.

For each scored item, the analysis records whether:

- Both conditions agree with the original score
- Only the 512-token condition agrees
- Only the 2,000-token condition agrees
- Neither condition agrees

The items for which only one condition agrees with the original score form the discordant pairs used in the exact paired test.

### Run the Analysis

```bash
python analysis/analyze_512_vs_2000_token_scores.py
```
```bash
python analysis/compare_token_limits_to_original_gpt4o.py
```
## Analysis Outputs

The analysis produces summary files for each comparison.

### 512-Token vs. 2,000-Token Comparison

- **Metric summary:** `analysis/AITA-NTA-FLIP_512_vs_2000_token_analysis.csv`

The metric summary reports the mean scores, mean differences, direction of label changes, and Exact McNemar/two-sided binomial sign test on discordant pairs results for validation, indirectness, and framing.


### Token limits vs. Original Comparison

- **Metric summary:** `analysis/AITA-NTA-FLIP_512_2000_vs_original_gpt4o_analysis.csv`

These files compare the scores produced under the 512-token and 2000-token condition with the corresponding scores released by the original study.

### Combined Results

- **Combined results table:** `../../figures/appendix_figures/full_max_token_analysis_table.pdf`
- **Analysis notebook:** `../../full_analysis/additional_graphs_appendix.ipynb`

The combined results summarize the direct token-limit comparison and both comparisons against the original released scores.

The maximum-token limit affected some individual scores—most notably framing—but neither condition consistently improved agreement with the original released results.
