
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Input data: Study hours and previous marks
X = np.array([
    [1, 40],
    [2, 45],
    [3, 50],
    [4, 60],
    [5, 70],
    [6, 80],
    [7, 90],
    [8, 95]
])

# Output labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create the classification model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Predict result for a new student
new_student = np.array([[6, 75]])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Predicted Result: Pass")
else:
    print("Predicted Result: Fail")

# Check accuracy on the training data
predicted_values = model.predict(X)
accuracy = accuracy_score(y, predicted_values)

print("Training Accuracy:", accuracy * 100, "%")
