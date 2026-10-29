
import pandas as pd
import numpy as np

# Create a dataset containing missing values and duplicates
data = {
    "Name": ["Amit", "Priya", "Rahul", "Priya"],
    "Age": [20, np.nan, 19, np.nan],
    "Marks": [85, 90, 78, 90]
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Handle missing values using the mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Remove duplicate records
df = df.drop_duplicates()

# Add a new column
df["Result"] = df["Marks"].apply(
    lambda x: "Pass" if x >= 40 else "Fail"
)

# Rename a column
df = df.rename(columns={"Marks": "Score"})

print("\nPreprocessed Dataset:\n", df)

# Check for remaining missing values
print("\nMissing Values:\n", df.isnull().sum())
