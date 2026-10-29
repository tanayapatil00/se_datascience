
import numpy as np

# Create two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Matrix operations
print("\nAddition:\n", A + B)
print("\nSubtraction:\n", A - B)
print("\nMatrix Multiplication:\n", np.dot(A, B))
print("\nTranspose of A:\n", A.T)

# Statistical operations
data = np.array([10, 20, 30, 40, 50])

print("\nMean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Standard Deviation:", np.std(data))
print("Minimum:", np.min(data))
print("Maximum:", np.max(data))
