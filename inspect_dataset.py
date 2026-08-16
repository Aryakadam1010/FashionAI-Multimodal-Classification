import os
import pandas as pd

# Read dataset
df = pd.read_csv(
    "dataset/styles.csv",
    engine="python",
    on_bad_lines="skip"
)

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n")

print("=" * 60)
print("UNIQUE MASTER CATEGORIES")
print("=" * 60)
print(df["masterCategory"].value_counts())

print("\n")

print("=" * 60)
print("TOP 20 ARTICLE TYPES")
print("=" * 60)
print(df["articleType"].value_counts().head(20))