import pandas as pd
import numpy as np

FILE_PATH = "/content/drive/MyDrive/Colab Notebooks/Data1(MATH509).xlsx"
SHEET_NAME = 0
OBSERVED_CUTOFF = "2024Q2"

def q_to_period(q):
    return pd.Period(str(q).strip().upper(), freq="Q")

# 1. Load first sheet only
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

# 2. Sort and create lagged predictors within each CMA
df["quarter_period"] = df["quarter"].apply(q_to_period)
df = df.sort_values(["cma", "quarter_period"]).copy()

base_macro = [
    "inflation",
    "disposable_income",
    "total_debt_payments",
    "mortgage_interest_paid",
    "bank_rate",
]

df["delinq_lag1"] = df.groupby("cma")["delinq_index_2012Q3_100"].shift(1)
for col in base_macro:
    df[f"{col}_lag1"] = df.groupby("cma")[col].shift(1)

features = ["delinq_lag1"] + [f"{c}_lag1" for c in base_macro]

observed_cutoff = pd.Period(OBSERVED_CUTOFF, freq="Q")
obs_df = df[df["quarter_period"] <= observed_cutoff].dropna(
    subset=["delinq_index_2012Q3_100"] + features
).copy()
