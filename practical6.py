
import pandas as pd

# Create a dataset
data = {
    "Department": [
        "Computer", "IT", "Computer",
        "IT", "Electronics", "Computer"
    ],
    "Student": [
        "Amit", "Priya", "Rahul",
        "Sneha", "Neha", "Raj"
    ],
    "Marks": [85, 90, 78, 88, 92, 80]
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Calculate average marks department-wise
print("\nAverage Marks by Department:\n",
      df.groupby("Department")["Marks"].mean())

# Calculate multiple statistics
print("\nDepartment-wise Statistics:\n",
      df.groupby("Department")["Marks"].agg(
          ["mean", "sum", "min", "max", "count"]
      ))

# Count students in each department
print("\nStudent Count:\n",
      df.groupby("Department")["Student"].count())
