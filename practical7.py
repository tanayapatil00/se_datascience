
import matplotlib.pyplot as plt

# Data for monthly sales
months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales = [100, 150, 120, 180, 200]

# 1. Line Plot
plt.figure()
plt.plot(months, sales, marker="o")
plt.title("Monthly Sales - Line Plot")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# 2. Bar Chart
plt.figure()
plt.bar(months, sales)
plt.title("Monthly Sales - Bar Chart")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# 3. Histogram
marks = [45, 55, 60, 65, 70, 72, 75, 80, 85, 90, 95]

plt.figure()
plt.hist(marks, bins=5, edgecolor="black")
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

