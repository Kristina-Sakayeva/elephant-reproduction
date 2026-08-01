# Judge Prompt Sensitivity

## Overview

This experiment evaluates whether differences between the judge prompts released in the original code and those printed in the paper appendix affect the resulting sycophancy scores.

The **indirectness** and **framing** prompts are largely consistent across the two versions, with differences mainly in formatting, grammar, wording, and output instructions.

The **validation** prompts differ more substantially. The code version uses a broader definition of emotional validation, including criteria related to:

- Acknowledging needs and desires
- Recognizing different perspectives
- Acknowledging strengths and weaknesses
- Recognizing situational difficulty
- Expressing warmth or concern
- Expressing similar feelings

The paper-appendix version compresses or omits several of these criteria. As a result, the code prompt may classify a wider range of borderline responses as emotionally validating.

## Research Question

**How sensitive are the judge scores to the version of the scoring prompt used?**

This experiment also examines whether differences between the code and appendix prompts can explain discrepancies between the reproduced judge scores and the originally released scores.

## Experimental Design

The same sample of previously generated model responses from the original paper was evaluated under two prompt conditions:

1. **Code-Prompt Condition:** Responses were scored using the judge prompts released with the original code.
2. **Paper-Prompt Condition:** The same responses were rescored using the judge prompts printed in the paper appendix.

The following dimensions were evaluated:

- **Validation**
- **Indirectness**
- **Framing**

Because the same responses and judge configuration were used in both conditions, differences between the resulting scores can be attributed to the version of the scoring prompt.

## Prompt Comparison

### Validation

The validation prompts contain the most substantive differences.

The code prompt defines emotional validation more broadly and includes several criteria that are shortened or omitted from the paper-appendix prompt. The code version may therefore classify more responses as validating.

### Indirectness

The indirectness prompts are semantically similar.

Most differences concern:

- Grammar
- Punctuation
- Output-format instructions
- Template placeholders
- The insertion of response values

### Framing

The framing prompts are nearly identical.

They preserve the same definition and label mapping, with differences mainly involving:

- Formatting
- Grammar
- Minor wording changes
- Small changes to examples

## Configuration

### Data Configuration

- **Dataset:** `AITA-YTA`
- **Sample size:** `50`
- **Response source:** `released responses`
- **Sampling method:** `random sample`
- **Response model:** `GPT-4o`

### Judge Configuration

- **Judge model:** `GPT-4o`
- **Model snapshot:** `gpt-4o-2024-11-20`
- **API provider:** `OpenAI`
- **Temperature:** `0`
- **Maximum output tokens:** `2`
- **Batching method:** `batch API`

### Prompt Conditions

#### Condition 1: Code Prompts

- **Prompt source:** Original code release
- **Prompt files:** `code/code_prompt_scorer.py`

#### Condition 2: Paper Prompts

- **Prompt source:** Original paper appendix
- **Prompt files:** `code/paper_prompt_scorer.py`

## Experimental Setup

### First Run: Condition 1

```bash
python code/code_prompt_scorer.py \
  --input_file sample/AITA-YTA-50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT-4o_code_prompt \
  --output_file results/GPT-4o_code_prompt_scoring.csv \
  --temperature 0
```

### Second Run: Condition 2

```bash
python code/paper_prompt_scorer.py \
  --input_file sample/AITA-YTA-50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT-4o_paper_prompt \
  --output_file results/GPT-4o_paper_prompt_scoring.csv \
  --temperature 0
```

## Analysis

For each sycophancy dimension, the analysis compares:

- The mean score under the code prompt
- The mean score under the paper prompt
- The number of responses assigned different labels
- The proportion of responses assigned different labels
- The direction of label changes
- The magnitude of aggregate score changes

Because the same responses are scored under both prompt conditions, the comparison is treated as a **paired analysis**.

## Original vs paper comparison:

```bash
python analysis/original_vs_paper_prompt.py \
  --scoring_csv results/GPT-4o_paper_prompt_scoring.csv \
  --output_dir analysis/original_vs_paper_prompt
```

For original vs code comparison, look in the temperature_sensitivity_self_consistency directory. For our comparison, we used the first temperature 0 run from temperature_sensitivity_self_consistency experiment.

## Outputs

- **Code-prompt scores:** `code/code_prompt_scorer.py`
- **Paper-prompt scores:** `code/paper_prompt_scorer.py`
- **Prompt comparison:** `analysis/original_vs_paper_prompt/metric_summary.csv and analysis/original_vs_code_prompt/full_combined_metric_summary_temp_sensitivity.csv`
- **Analysis notebook:** `../../full_analysis/additional_graphs_appendix.ipynb`
- **Figure:** `../../figures/appendix_figures/prompt_version_mean_comparison_table.pdf`

## Interpretation

Differences in prompt wording may introduce variation into LLM-based evaluation, particularly when one prompt contains a broader or more detailed scoring rubric.

This experiment tests whether the differences between the code and paper-appendix prompts meaningfully affect judge outputs and whether those differences explain discrepancies from the originally released scores.

The experiment is not intended to determine that one prompt version is inherently more correct than the other.
