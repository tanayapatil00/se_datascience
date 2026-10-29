
import pandas as pd

# Create a dataset
data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Neha"],
    "Age": [20, 21, 19, 22, 20],
    "Marks": [85, 90, 78, 92, 88]
}

df = pd.DataFrame(data)

# Save dataset to a CSV file
df.to_csv("students.csv", index=False)

# Import dataset from CSV file
df = pd.read_csv("students.csv")

# Display the dataset
print("Dataset:\n", df)

# Display first five rows
print("\nFirst Five Rows:\n", df.head())

# Display dataset information
print("\nDataset Information:")
df.info()

# Display number of rows and columns
print("\nNumber of Rows and Columns:", df.shape)

# Display column names
print("\nColumn Names:", df.columns.tolist())

# Display descriptive statistics
print("\nDescriptive Statistics:\n", df.describe())
