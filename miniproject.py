# ============================================================
# MINI PROJECT:
# STUDENT PERFORMANCE ANALYSIS AND PREDICTION USING PYTHON
#
# Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 65)
print(" STUDENT PERFORMANCE ANALYSIS AND PREDICTION")
print("=" * 65)


# ------------------------------------------------------------
# PRACTICAL 1: NUMPY ARRAYS AND VECTORIZED OPERATIONS
# ------------------------------------------------------------

print("\n1. NUMPY ARRAY OPERATIONS")

marks_array = np.array([65, 78, 82, 45, 91, 56, 73, 88])

print("Original Marks:", marks_array)
print("First Three Marks:", marks_array[:3])
print("Marks Above 75:", marks_array[marks_array > 75])
print("Marks After Adding 5:", marks_array + 5)
print("Reshaped Array:\n", marks_array.reshape(2, 4))


# ------------------------------------------------------------
# PRACTICAL 2: MATRIX AND STATISTICAL OPERATIONS
# ------------------------------------------------------------

print("\n2. MATRIX AND STATISTICAL OPERATIONS")

matrix = np.array([
    [65, 70, 75],
    [80, 85, 90],
    [55, 60, 65]
])

print("Marks Matrix:\n", matrix)
print("Transpose:\n", matrix.T)
print("Matrix Addition:\n", matrix + 5)
print("Column Averages:", np.mean(matrix, axis=0))
print("Overall Mean:", np.mean(matrix))
print("Overall Standard Deviation:", np.std(matrix))


# ------------------------------------------------------------
# PRACTICAL 3: CREATE AND INSPECT DATASET USING PANDAS
# ------------------------------------------------------------

print("\n3. DATASET CREATION AND INSPECTION")

np.random.seed(42)

number_of_students = 100

study_hours = np.random.uniform(
    0.5, 8, number_of_students
)

attendance = np.random.uniform(
    45, 100, number_of_students
)

previous_marks = np.random.uniform(
    30, 95, number_of_students
)

# Generate marks influenced by study hours,
# attendance, previous marks, and random variation.

noise = np.random.normal(
    0, 6, number_of_students
)

final_marks = (
    15
    + 4.5 * study_hours
    + 0.12 * attendance
    + 0.45 * previous_marks
    + noise
)

final_marks = np.clip(final_marks, 0, 100)

df = pd.DataFrame({
    "Study_Hours": study_hours.round(2),
    "Attendance": attendance.round(2),
    "Previous_Marks": previous_marks.round(2),
    "Final_Marks": final_marks.round(2)
})

# Introduce a few missing values for preprocessing practice.

df.loc[5, "Attendance"] = np.nan
df.loc[12, "Study_Hours"] = np.nan
df.loc[20, "Previous_Marks"] = np.nan

# Add a duplicate row to demonstrate duplicate removal.

df = pd.concat(
    [df, df.iloc[[0]]],
    ignore_index=True
)

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())


# ------------------------------------------------------------
# PRACTICAL 4: DATA PREPROCESSING
# ------------------------------------------------------------

print("\n4. DATA PREPROCESSING")

print("Missing Values Before Cleaning:")
print(df.isnull().sum())

print("Duplicate Rows Before Cleaning:", df.duplicated().sum())

# Remove duplicate records.

df.drop_duplicates(inplace=True)

# Fill missing numerical values with column medians.

df.fillna(df.median(numeric_only=True), inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("Duplicate Rows After Cleaning:", df.duplicated().sum())

# Add a categorical performance column.

df["Performance"] = np.where(
    df["Final_Marks"] >= 40,
    "Pass",
    "Fail"
)

# Add study-hour categories.

df["Study_Category"] = pd.cut(
    df["Study_Hours"],
    bins=[0, 2, 5, 8],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

print("\nCleaned Dataset:")
print(df.head())


# ------------------------------------------------------------
# PRACTICAL 5: FILTERING, SORTING AND CONDITIONAL SELECTION
# ------------------------------------------------------------

print("\n5. FILTERING AND SORTING")

high_performers = df[df["Final_Marks"] >= 75]

print("\nStudents Scoring 75 or More:")
print(high_performers.head())

print("\nTop 10 Students:")
print(
    df.sort_values(
        by="Final_Marks",
        ascending=False
    ).head(10)
)

print("\nStudents With Attendance Above 75%:")
print(
    df[df["Attendance"] > 75][
        ["Attendance", "Final_Marks"]
    ].head()
)


# ------------------------------------------------------------
# PRACTICAL 6: GROUPING AND AGGREGATION
# ------------------------------------------------------------

print("\n6. GROUPING AND AGGREGATION")

group_summary = df.groupby(
    "Study_Category",
    observed=True
).agg(
    Average_Marks=("Final_Marks", "mean"),
    Maximum_Marks=("Final_Marks", "max"),
    Minimum_Marks=("Final_Marks", "min"),
    Student_Count=("Final_Marks", "count")
)

print("\nPerformance by Study Category:")
print(group_summary.round(2))

attendance_summary = df.groupby(
    "Performance",
    observed=True
).agg(
    Average_Attendance=("Attendance", "mean"),
    Average_Study_Hours=("Study_Hours", "mean"),
    Average_Marks=("Final_Marks", "mean")
)

print("\nPerformance Group Summary:")
print(attendance_summary.round(2))


# ------------------------------------------------------------
# PRACTICAL 7: LINE PLOT, BAR CHART AND HISTOGRAM
# ------------------------------------------------------------

print("\n7. BASIC DATA VISUALIZATION")

# Line plot: marks of the first 20 students.

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, 21),
    df["Final_Marks"].iloc[:20],
    marker="o"
)

plt.title("Final Marks of First 20 Students")
plt.xlabel("Student Number")
plt.ylabel("Final Marks")
plt.grid(True)
plt.tight_layout()
plt.show()


# Bar chart: average marks by study category.

group_summary["Average_Marks"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Average Marks by Study Category")
plt.xlabel("Study Category")
plt.ylabel("Average Marks")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Histogram: distribution of marks.

plt.figure(figsize=(8, 5))

plt.hist(
    df["Final_Marks"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PRACTICAL 8: SCATTER PLOTS AND CORRELATION HEATMAP
# ------------------------------------------------------------

print("\n8. RELATIONSHIP VISUALIZATION")

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Marks",
    hue="Performance"
)

plt.title("Study Hours vs Final Marks")
plt.tight_layout()
plt.show()


# Correlation matrix for numerical columns.

numerical_columns = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Final_Marks"
]

correlation_matrix = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PRACTICAL 9: STATISTICAL ANALYSIS
# ------------------------------------------------------------

print("\n9. STATISTICAL ANALYSIS")

for column in numerical_columns:
    print(f"\nStatistics for {column}")

    print("Mean:", round(df[column].mean(), 2))
    print("Median:", round(df[column].median(), 2))
    print("Variance:", round(df[column].var(), 2))
    print("Standard Deviation:", round(df[column].std(), 2))

print("\nCorrelation Between Study Hours and Final Marks:")

print(
    round(
        df["Study_Hours"].corr(df["Final_Marks"]),
        3
    )
)


# ------------------------------------------------------------
# PRACTICAL 10: LINEAR REGRESSION
# PREDICT FINAL MARKS
# ------------------------------------------------------------

print("\n10. LINEAR REGRESSION")

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks"
]

X = df[features]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

predicted_marks = regression_model.predict(X_test)

print("\nActual Marks:")
print(y_test.head().round(2).to_list())

print("\nPredicted Marks:")
print(np.round(predicted_marks[:5], 2).tolist())

mae = mean_absolute_error(y_test, predicted_marks)

mse = mean_squared_error(y_test, predicted_marks)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predicted_marks)

print("\nRegression Evaluation:")
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("R2 Score:", round(r2, 3))


# ------------------------------------------------------------
# PRACTICAL 11: CLASSIFICATION
# PREDICT PASS OR FAIL
# ------------------------------------------------------------

print("\n11. CLASSIFICATION")

# The pass/fail label is created from final marks.
# It is not included in the classifier's input features.

X_classification = df[features]

y_classification = np.where(
    df["Final_Marks"] >= 40,
    1,
    0
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_classification,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

classifier.fit(X_train_c, y_train_c)

classification_predictions = classifier.predict(X_test_c)

print("\nClassification Accuracy:",
      round(
           accuracy_score(
              y_test_c,
              classification_predictions
          ) * 100,
          2
      ),
      "%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test_c,
        classification_predictions,
        labels=[0, 1],
        target_names=["Fail", "Pass"],
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test_c,
    classification_predictions,
    labels=[0, 1]
)

print("Confusion Matrix:")
print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.title("Pass/Fail Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PRACTICAL 12: MINI PROJECT - PREDICT A NEW STUDENT'S MARKS
# ------------------------------------------------------------

print("\n12. MINI PROJECT: NEW STUDENT PREDICTION")

new_student = pd.DataFrame({
    "Study_Hours": [5.0],
    "Attendance": [85.0],
    "Previous_Marks": [75.0]
})

predicted_final_marks = regression_model.predict(
    new_student[features]
)[0]

predicted_category = (
    "Pass" if predicted_final_marks >= 40 else "Fail"
)

predicted_class = classifier.predict(
    new_student[features]
)[0]

print("\nNew Student Details:")
print(new_student.to_string(index=False))

print(
    "\nPredicted Final Marks:",
    round(float(predicted_final_marks), 2)
)

print("Category Based on Predicted Marks:", predicted_category)

print(
    "Classifier Prediction:",
    "Pass" if predicted_class == 1 else "Fail"
)

print("\nProject Completed Successfully!")

print("=" * 65)