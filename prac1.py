import numpy as np


# ====================================================================
# 1. ARRAY CREATION
# ====================================================================

print("--- 1. ARRAY CREATION ---")

l1 = [10, 20, 30, 40, 50, 60]

# Creating 1D and 2D arrays
arr_1d = np.array(l1)

arr_2d = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [11, 12, 23, 67]
])

print("1D Array:", arr_1d)
print("2D Array:\n", arr_2d)
print()


# ====================================================================
# 2. INDEXING
# Accessing individual elements
# ====================================================================

print("--- 2. INDEXING ---")

first_element = arr_1d[0]
last_element = arr_1d[-1]

element_2d = arr_2d[1, 2]    # Row index 1, Column index 2 (value: 7)

print(f"First element of 1D: {first_element}")
print(f"Last element of 1D: {last_element}")
print(f"Element at row 1, column 2 in 2D: {element_2d}")
print()

# # # # #


# ====================================================================
# 3. SLICING
# Extracting sub-arrays [start:stop:step]
# ====================================================================

print("--- 3. SLICING ---")

slice_1d = arr_1d[1:4]          # Elements from index 1 up to (but not including) 4
sub_grid = arr_2d[:2, 1:3]      # Top 2 rows, middle 2 columns

print("1D Slice [1:4]:", slice_1d)
print("2D Sub-grid (Rows 0-1, Cols 1-2):\n", sub_grid)
print()

# # # # #


# ====================================================================
# 4. VECTORIZED OPERATIONS
# Element-wise operations performed without loops
# ====================================================================

print("--- 4. VECTORIZED OPERATIONS ---")

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

addition = a + b                  # Vectorized addition
multiplication = a * 5            # Scalar multiplication
squared = a ** 2                  # Element-wise squaring
sine_values = np.sin(a)           # Vectorized trigonometric function

print("Vectorized Addition (a + b):", addition)
print("Scalar Multiplication (a * 5):", multiplication)
print("Element-wise Power (a ** 2):", squared)
print("Sine values of 'a':", sine_values)
print()

# # # # #


# ====================================================================
# 5. BOOLEAN INDEXING (FILTERING)
# Vectorized condition check to filter elements
# ====================================================================

print("--- 5. BOOLEAN INDEXING (FILTERING) ---")

prices = np.array([15, 80, 45, 120, 30, 95])

expensive_prices = prices[prices > 50]    # Extracts elements greater than 50

print("Original Prices:", prices)
print("Prices > 50:", expensive_prices)