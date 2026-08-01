# Judge Temperature Sensitivity and Self-Consistency

## Overview

This experiment evaluates whether the sampling temperature used for GPT-4o judge scoring affects the reproducibility of binary sycophancy labels.

The original study did not report the temperature used during judge scoring. To test whether this undocumented setting could affect the released results, the same sample of responses was rescored under two temperature configurations:

1. **API default temperature**
2. **Temperature 0**

The temperature-0 condition was repeated three times to evaluate run-to-run self-consistency.

All runs used the same model responses, scoring prompts, judge model, and batch-based evaluation procedure.

## Research Questions

This experiment addresses two questions:

1. **How sensitive are judge scores to the temperature used during evaluation?**
2. **How consistent are repeated GPT-4o judge evaluations at temperature 0?**

The analysis also compares each reproduced run with the corresponding scores released by the original study.

## Experimental Design

A random sample of GPT-4o responses from the released **AITA-YTA** results was selected.

Each response was scored for:

- **Validation**
- **Indirectness**
- **Framing**

The experiment included four judge runs:

1. **One run using the API default temperature**
2. **Temperature-0 Run 1**
3. **Temperature-0 Run 2**
4. **Temperature-0 Run 3**

The following were held constant across all runs:

- Sampled responses
- Scoring prompts
- Judge model
- Model snapshot
- Batch-based scoring procedure

Because the same responses were evaluated in every run, all comparisons are treated as paired analyses.

## Experimental Conditions

### Condition 1: Default Temperature

The sampled responses were scored using the API’s default temperature setting.

This condition evaluates the behavior of the judge when no explicit temperature is supplied.

### Condition 2: Temperature 0

The same responses were scored with the judge temperature explicitly set to `0`.

This condition was repeated three times to measure whether temperature 0 produced identical labels across repeated evaluations.

## Configuration

### Data Configuration

- **Dataset:** `AITA-YTA`
- **Sample size:** `50`
- **Sampling method:** `random sample`
- **Response source:** `released responses`
- **Response model:** `GPT-4o`
- **Sample file:** `sample/AITA-YTA_50_sample_complete.csv`

### Judge Configuration

- **Judge model:** `GPT-4o`
- **Model snapshot:** `gpt-4o-2024-11-20`
- **API provider:** `OpenAI`
- **Evaluation method:** `batch API`
- **Maximum output tokens:** `2`
- **Dimensions evaluated:**
  - Validation
  - Indirectness
  - Framing
- **Output format:** `csv`

### Temperature Conditions

#### Default-Temperature Run

- **Temperature:** `API default`
- **Number of runs:** `1`

#### Temperature-0 Runs

- **Temperature:** `0`
- **Number of runs:** `3`

## Experimental Setup

### Run the Default-Temperature Condition

```bash
python code/sycophancy_scorers_consistency_check.py \
  --input_file sample/AITA-YTA_50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT4o_default_run1 \
  --output_file results/test1_default_run1.csv
```

The temperature argument is omitted so that the API default is used.

### Run Temperature-0 Evaluation 1

```bash
python code/sycophancy_scorers_consistency_check.py \
  --input_file sample/AITA-YTA_50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT4o_temp0_run1 \
  --output_file results/test2_temp0_run1.csv \
  --temperature 0
```

### Run Temperature-0 Evaluation 2

```bash
python code/sycophancy_scorers_consistency_check.py \
  --input_file sample/AITA-YTA_50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT4o_temp0_run4 \
  --output_file results/test2_temp0_run4.csv \
  --temperature 0
```

### Run Temperature-0 Evaluation 3

```bash
python code/sycophancy_scorers_consistency_check.py \
  --input_file sample/AITA-YTA_50_sample_complete.csv \
  --prompt_column prompt \
  --response_column GPT-4o \
  --output_column_tag GPT4o_temp0_run5 \
  --output_file results/test2_temp0_run5.csv \
  --temperature 0
```

## Analysis

The analysis contains three main components:

1. **Default temperature vs. temperature 0**
2. **Each reproduced run vs. the original released scores**
3. **Self-consistency across the three temperature-0 runs**

### Default Temperature vs. Temperature 0

The default-temperature run is compared with the first temperature-0 run.

For each dimension, the analysis calculates:

- Mean score under the default temperature
- Mean score under temperature 0
- Mean-score difference
- Number of observations with scores available in both runs
- Number of disagreements
- Disagreement rate
- Direction of label changes

The disagreement rate is calculated only over observations for which both compared labels are available.

### Comparison With the Released Scores

The following reproduced runs are compared independently with the original study’s released judge scores:

- Default-temperature run
- Temperature-0 Run 1
- Temperature-0 Run 2
- Temperature-0 Run 3

For each comparison and dimension, the analysis calculates:

- Number of available paired labels
- Number of disagreements
- Disagreement rate
- Reproduced mean score
- Released mean score
- Mean-score difference

Rows with blank released labels are excluded from the corresponding paired comparison.

### Temperature-0 Self-Consistency

The three temperature-0 runs are compared to determine whether repeated evaluations produce identical labels.

For each dimension, the analysis reports:

- Mean score for Run 1
- Mean score for Run 2
- Mean score for Run 3
- Number of self-disagreements
- Self-disagreement rate

A response is counted as a self-disagreement when the three temperature-0 runs do not all assign the same binary label.

Identical aggregate means do not necessarily imply perfect response-level consistency because label changes in opposite directions may cancel out.

## Run the Analysis

```bash
python analysis/analysis_default_vs_temp0.py \
  --default_csv results/test1_default_run1.csv \
  --temp0_csv results/test2_temp0_run1.csv \
  --output_dir analysis/default_vs_temp0

python analysis/original_vs_default.py \
  --original_csv sample/AITA-YTA_50_sample_complete.csv \
  --default_csv results/test1_default_run1.csv \
  --output_dir analysis/original_vs_default

python analysis/original_vs_temp0.py \
  --original_csv sample/AITA-YTA_50_sample_complete.csv \
  --run1_csv results/test2_temp0_run1.csv \
  --run4_csv results/test2_temp0_run4.csv \
  --run5_csv results/test2_temp0_run5.csv \
  --output_dir analysis/original_vs_temp0

python analysis/self_consistency_temp0.py \
  --run1_csv results/test2_temp0_run1.csv \
  --run2_csv results/test2_temp0_run4.csv \
  --run3_csv results/test2_temp0_run5.csv \
  --output_dir analysis/self_consistency_check_temp0

python analysis/combine_analysis_summaries.py
```
Note: Due to API-side errors, temperature-0 Runs 4 and 5 are used as the second and third successful runs in the analysis.


## Analysis Outputs

The analysis produces outputs for:

- **Default temperature vs. temperature-0 Run 1**
- **Default temperature vs. released scores**
- **Temperature-0 Run 1 vs. released scores**
- **Temperature-0 Run 2 vs. released scores**
- **Temperature-0 Run 3 vs. released scores**
- **Self-consistency across all temperature-0 runs**
- **Individual disagreement cases**

## Outputs

- **Sampled responses:** `sample/AITA-YTA_50_sample_complete.csv`
- **Default-temperature scores:** `results/test1_default_run1.csv`
- **Temperature-0 Run 1 scores:** `results/test2_temp0_run1.csv`
- **Temperature-0 Run 2 scores:** `results/test2_temp0_run4.csv`
- **Temperature-0 Run 3 scores:** `results/test2_temp0_run5.csv`
- **Comparison summary:** `analysis/combined_metric_summary.csv`
- **Analysis notebook:** `../../full_analysis/additional_graphs_appendix.ipynb`
- **Disagreement-rate figure:** `../../figures/appendix_figures/judge_disagreement_rates_table.pdf`
- **Self-consistency figure:** `../../figures/appendix_figures/judge_self_consistency_table.pdf`

## Interpretation

The experiment evaluates both aggregate score stability and response-level reproducibility.

The default-temperature and temperature-0 conditions may produce similar mean scores while still disagreeing on individual labels. Repeating the temperature-0 condition tests whether explicitly setting the temperature reduces this run-to-run variation.

Comparisons with the released scores evaluate whether temperature choice explains discrepancies between the reproduced judge evaluations and the original results.

## Conclusion

Temperature 0 produced highly consistent repeated judge evaluations, although validation was not perfectly deterministic, and it yielded slightly closer agreement with the released validation and framing scores than the default-temperature condition.
