import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import mean_squared_error

# Read data from file
file_path = "Group_1_Data.txt"  # Replace with your file path
x = []
f_x = []

with open(file_path, 'r') as file:
    next(file)  # Skip header line
    for line in file:
        values = line.strip().split(',')
        x.append(float(values[0]))
        f_x.append(float(values[2]))

x = np.array(x)
f_x = np.array(f_x)

# Polynomial fitting (degree can be tuned)
poly_degree = 5  # Change degree as necessary
coeffs = np.polyfit(x, f_x, poly_degree)
poly = np.poly1d(coeffs)

# Predicted values
x_pred = np.linspace(min(x), max(x), 200)
f_pred = poly(x_pred)

# Calculate Mean Squared Error
mse = mean_squared_error(f_x, poly(x))
print(f"Mean Squared Error (degree {poly_degree}): {mse:.5f}")

# Plot original data and fitted polynomial
plt.scatter(x, f_x, color='red', label='Data Points')
plt.plot(x_pred, f_pred, label=f'Polynomial Fit (degree {poly_degree})', color='blue')
plt.title('Approximating f(x, y) with Polynomial Regression')
plt.xlabel('x')
plt.ylabel('f(x, y)')
plt.legend()
plt.grid()
plt.show()
