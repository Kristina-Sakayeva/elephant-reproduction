import pandas as pd

df = pd.read_csv("elephant_full_results/AITA-YTA_full_results.csv")

sample = df.sample(n=50, random_state=42).copy()
sample["sample_id"] = sample.index

sample.to_csv("elephant/LLM_judge_self_consistency_check/aita_yta_50_sample_full_paper_responses.csv", index=False)
