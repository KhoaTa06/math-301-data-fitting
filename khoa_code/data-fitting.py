import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations_with_replacement
from scipy.optimize import curve_fit


def fit_function(df, degree, sample_size=10100):
    data_point_index = 0
    # Extract x, y, and f(x,y) values from DataFrame
    if df.size > sample_size:
        df = df.sample(sample_size, random_state=42)

    x, y, fxy = df['x'].values, df['y'].values, df['fxy'].values
    
    # Generate polynomial terms up to degree 12
    def poly_func(X, *coeffs):
        x, y = X
        terms = [x**i * y**j for i, j in combinations_with_replacement(range(degree+1), 2)]
        return sum(c * t for c, t in zip(coeffs, terms))
    
    # Fit the function to the data
    num_coeffs = sum(1 for _ in combinations_with_replacement(range(degree+1), 2))
    initial_guess = np.ones(num_coeffs)
    params, _ = curve_fit(poly_func, (x, y), fxy, p0=initial_guess, maxfev=sample_size*3)
    
    # Return the function with the fitted parameters
    def fitted_function(x, y):
        return poly_func((x, y), *params)
    
    return fitted_function, params


def fit_polynomial_degree(df, max_degree=50):
    X = df[['x', 'y']].values
    y = df['fxy'].values
    
    best_degree = 1
    best_score = -np.inf  # For maximizing R²
    best_model = None

    results = []

    for degree in range(1, max_degree + 1):
        print("calculating: ", degree)
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        results.append((degree, r2, rmse))

        if r2 > best_score:
            best_score = r2
            best_degree = degree
            best_model = model

    # Convert results to DataFrame for better visualization
    results_df = pd.DataFrame(results, columns=['Degree', 'R² Score', 'RMSE'])

    return best_degree, best_model, results_df


def test_fitting_accuracy(data, predict_data, fitted_func):
    y_true = data['fxy'].values
    y_pred = np.array([fitted_func(x, y) for x, y in zip(data['x'], data['y'])])
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    predict_data['fxy'] = y_pred
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    return mse, r2


def interactive_3d_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=df['x'], y=df['y'], z=df['fxy'], mode='markers', marker=dict(size=2, color=df['fxy'], colorscale='Viridis')))
    
    fig.update_layout(title='Interactive 3D Plot of Data Points',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(X, Y)'))
    fig.show()


def save_fitted_function(params, filename="fitted_params.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(params, f)


def load_fitted_function(filename="fitted_params.pkl", degree=5):
    with open(filename, "rb") as f:
        params = pickle.load(f)

    def poly_func(X, *coeffs):
        x, y = X
        terms = [x**i * y**j for i, j in combinations_with_replacement(range(degree + 1), 2)]
        return sum(c * t for c, t in zip(coeffs, terms))

    def fitted_function(x, y):
        return poly_func((x, y), *params)

    return fitted_function


dataFilepath = "Group_1_Data.txt"
data = pd.read_csv(dataFilepath)
data.columns
data.drop('y)', axis=1, inplace=True)
data.columns=["x", "y", "fxy"]

predict_data = pd.read_csv(dataFilepath)
predict_data.columns
predict_data.drop('y)', axis=1, inplace=True)
predict_data.columns=["x", "y", "fxy"]

# best_degree, best_model, results_df = fit_polynomial_degree(data)
# print("Best polynomial degree:", best_degree)
# print(results_df)

best_degree = 12

print("data size: ", len(data))

fitted_func, params = fit_function(data, best_degree, sample_size=30000)
print("Fitted parameters:", params)
save_fitted_function(params)

fitted_func = load_fitted_function(degree=best_degree)

test_fitting_accuracy(data, predict_data, fitted_func)

interactive_3d_plot(data)

interactive_3d_plot(predict_data)