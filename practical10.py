
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Input data: Study hours
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Output data: Student marks
y = np.array([35, 40, 50, 60, 70])

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict marks for 6 hours of study
prediction = model.predict([[6]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted Marks:", prediction[0])

# Evaluate the model
predicted_values = model.predict(X)
mse = mean_squared_error(y, predicted_values)

print("Mean Squared Error:", mse)
