
import pandas as pd

# Create a dataset
data = {
    "Name": ["Amit", "Priya", "Rahul", "Sneha", "Neha"],
    "Marks": [75, 92, 35, 88, 65],
    "Age": [20, 21, 19, 22, 20]
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Filter students scoring above 70
print("\nStudents Scoring Above 70:\n",
      df[df["Marks"] > 70])

# Select students with marks >= 40 and age >= 20
print("\nStudents Who Passed:\n",
      df[(df["Marks"] >= 40) & (df["Age"] >= 20)])

# Sort students by marks in descending order
print("\nSorted Dataset:\n",
      df.sort_values(by="Marks", ascending=False))

# Select specific columns
print("\nNames and Marks:\n",
      df[["Name", "Marks"]])
