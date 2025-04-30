import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from scipy.optimize import curve_fit

from testing_error import test_fitting_accuracy, calculate_slice_error
from load_graph import interactive_3d_plot, plot_y_slice, plot_x_slice
from derivatives import compute_derivatives


def save_fitted_function(params, filename="fitted_params.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def load_fitted_function(filename="fitted_params_50000.pkl", degree=5):
    with open(filename, "rb") as f:
        params = pickle.load(f)

    def poly_func(X, *coeffs):
        x, y = X
        terms = [x**i * y**j for i, j in combinations_with_replacement(range(degree + 1), 2)]
        return sum(c * t for c, t in zip(coeffs, terms))

    def fitted_function(x, y):
        return poly_func((x, y), *params)

    return fitted_function


def fit_slices(df):
    unique_y = df['y'].unique()
    slice_fits = {}
    
    def sin_func(x, A, B, C):
        return A * np.sin(B * x) + C
    
    for y_val in unique_y:
        slice_df = df[df['y'] == y_val]
        x, fxy = slice_df['x'].values, slice_df['fxy'].values
        
        if len(x) < 3:
            continue  # Not enough points to fit
        
        initial_guess = [1, 1, 1]
        params, _ = curve_fit(sin_func, x, fxy, p0=initial_guess, maxfev=10000)
        slice_fits[y_val] = params
    
    return slice_fits


# def function_template(xy, a, b, c, d, e, f, k): MSE:1.798
#     x, y = xy
#     x_term = np.power(x + 1e-6, k) / (1 + c * x)
#     return a * x_term * np.sin(b * y + d) + e * y + f


# def function_template(xy, a, b, c, d, e, f, g, h, i, k): MSE: 0.6439
#     x, y = xy
#     x_term = np.power(x + 1e-6, k) / (1 + c * x)
#     return (a * x_term * np.sin(b * y + d) +
#             g * x_term * np.sin(2 * b * y + h) +
#             e * y + i * y**2 + f)


# MSE: 0.6438
def x_func_template(x, a, b, c, d, e):
    return a * x + b * np.log(c * x + d + 1e-6) + e

def y_func_template(y, a, b, c, d, e, f):
    return a * np.sin(b * y + c) + d * np.log(e * y + 1) + f

def function_template(xy, a, b, c, d, e, f, g, h, i, j, k):
    x, y = xy
    # x_term = np.power(x + 1e-6, k) / (1 + c * x)
    # return (a * x_term * np.sin(b * y + d) +
    #         g * x_term * np.sin(2 * b * y + h) +
    #         j * x_term * np.sin(3 * b * y + l) +
    #         e * y + i * y**2 + f)

    x_func = x_func_template(x, i, j, a, c, k)
    y_func = y_func_template(y, d, b, f, e, h, g)
    return x_func * y_func


def function_template_noise(xy, a, b, c, d, e, f, g, h, i, j, k, sigma=0.03):
    x, y = xy
    # x_term = np.power(x + 1e-6, k) / (1 + c * x)
    # fxy = (a * x_term * np.sin(b * y + d) +
    #        g * x_term * np.sin(2 * b * y + h) +
    #        j * x_term * np.sin(3 * b * y + l) +
    #        e * y + i * y**2 + f)
    
    # # Add Gaussian noise
    # noise = np.random.normal(0, noise_std, size=x.shape if isinstance(x, np.ndarray) else 1)
    # return fxy + noise

    x_func = x_func_template(x, i, j, a, c, k)
    y_func = y_func_template(y, d, b, f, e, h, g)
    fxy = x_func * y_func
    
    # Add Gaussian noise with fitted sigma
    noise = np.random.normal(0, sigma, size=x.shape if isinstance(x, np.ndarray) else 1)
    return fxy + noise


def function_with_multiplicative_noise(xy, a, b, noise_std=0.1):
    x, y = xy
    # x_term = np.power(x + 1e-6, k) / (1 + c * x)
    # fxy = (a * x_term * np.sin(b * y + d) +
    #        g * x_term * np.sin(2 * b * y + h) +
    #        j * x_term * np.sin(3 * b * y + l) +
    #        e * y + i * y**2 + f)
    
    # Add multiplicative Gaussian noise
    # noise = np.random.normal(0, noise_std, size=x.shape if isinstance(x, np.ndarray) else 1)
    # return fxy * (1 + noise)

    log_term = np.log(a * (x + 1e-6))
    fxy = log_term * np.sin(b * y) + y
    
    # Add Gaussian noise with fitted sigma
    noise = np.random.normal(0, noise_std, size=x.shape if isinstance(x, np.ndarray) else 1)
    return fxy + noise


def generate_fxy_dataframe(data, params):
    # Validate parameters
    required_params = {'a', 'b'}
    if not all(param in params for param in required_params):
        missing = required_params - set(params.keys())
        raise ValueError(f"Missing required parameters: {missing}")
    
    # Load data if a file path is provided
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    # Ensure required columns exist
    if not all(col in data.columns for col in ['x', 'y']):
        raise ValueError("DataFrame must contain 'x' and 'y' columns")
    
    # Prepare input for fxy_func
    xy = (data['x'].values, data['y'].values)
    
    # Calculate f(x,y) using the predefined function
    fxy_calculated = function_with_multiplicative_noise(xy, params['a'], params['b'])
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = data.copy()
    result_df['fxy'] = fxy_calculated
    
    return result_df

dataFilepath = "/Users/khoa/Desktop/Math_301/math-301-data-fitting/khoa_code/Group_1_Data_10000.txt"
data = pd.read_csv(dataFilepath)
data.columns
data.drop('y)', axis=1, inplace=True)
data.columns=["x", "y", "fxy"]

predict_data = pd.read_csv(dataFilepath)
predict_data.columns
predict_data.drop('y)', axis=1, inplace=True)
predict_data.columns=["x", "y", "fxy"]

best_degree = 12

# FIND SIN POLYNOMIAL DEGREE
# best_sin_poly_degree = fit_polynomial_degree_sin_poly(data, predict_data, max_degree=25, sample_size=30000)
# print("Best sin(polynomial) degree:", best_sin_poly_degree)
#END
print("data size: ", len(data))

# plot_slice(0.0000, data, slice_fits[0])

# mse, r2 = calculate_slice_error(y_value=0.0000, df=data, slice_fits=slice_fits)

# Prepare data for curve_fit
x_data = data['x'].values
y_data = data['y'].values
fxy_data = data['fxy'].values
xy_data = np.vstack((x_data, y_data))

# Initial parameter guesses
p0 = [7, 4.3, 0.002, 2.1, 0.05, -0.7, 1.5, 2, 0.5, 0.5, 0.1]
# Fit the function
try:
    popt, pcov = curve_fit(function_template, xy_data, fxy_data, p0=p0, maxfev=10000)
    a, b, c, d, e, f, g, h, i, j, k = popt
    print(f"""Fitted parameters: 
        a={a:.3f},
        b={b:.3f},
        c={c:.3f},
        d={d:.3f},
        e={e:.3f},
        f={f:.3f},
        g={g:.3f},
        h={h:.3f},
        i={i:.3f},
        j={j:.3f},
        k={k:.3f}
            """)
except RuntimeError as e:
    print(f"Fitting failed: {e}")

# Evaluate the refined fit
fitted_refined = function_template_noise(xy_data, *popt)
residuals_refined = fxy_data - fitted_refined
mse_refined = np.mean(residuals_refined**2)
print(f"Refined Mean Squared Error: {mse_refined:.6f}")

print("Final function: ")

predict_data = generate_fxy_dataframe(predict_data, {
    'a': a, 'b': b, 'c': c, 'd': d,
    'e': e, 'f': f, 'g': g, 'h': h,
    'i': i, 'j': j, 'k': k
})


# Derivatives
derivative_data = compute_derivatives(data)


# plot_y_slice(data, derivative_data)
plot_x_slice(data, data)

# test_fitting_accuracy(data, predict_data, fitted_func)

# interactive_3d_plot(data, derivative_data)

