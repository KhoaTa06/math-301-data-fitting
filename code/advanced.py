import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the dataset
def load_data(Group_1_Data.txt):
    """
    Load the dataset from a .txt file.
    Assumes the file has columns x, y, and f(x, y) separated by commas.
    """
    data = pd.read_csv(Group_1_Data.txt, header=None, names=['x1', 'x2', 'y'])
    X = data[['x1', 'x2']].values  # First two columns are x1 and x2
    y = data['y'].values           # Last column is y
    return X, y

# Split the dataset into training and testing sets
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Linear Regression
def linear_regression(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    return mse, mae, r2

# Polynomial Regression
def polynomial_regression(X_train, X_test, y_train, y_test, degree=2):
    """
    Train and evaluate a polynomial regression model.
    """
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Polynomial Regression (Degree {degree}) Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    return mse, mae, r2

# Neural Network
def neural_network(X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
    """
    Train and evaluate a simple neural network model.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)  # Output layer
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred = model.predict(X_test).flatten()
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Neural Network Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2: {r2:.4f}")
    return mse, mae, r2

# Main function
def main():
    # Load and split the data
    file_path = "data.txt"  # Replace with your dataset file path
    X, y = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Evaluate models
    print("\nEvaluating Models...\n")
    lr_mse, lr_mae, lr_r2 = linear_regression(X_train, X_test, y_train, y_test)
    pr_mse, pr_mae, pr_r2 = polynomial_regression(X_train, X_test, y_train, y_test, degree=2)
    nn_mse, nn_mae, nn_r2 = neural_network(X_train, X_test, y_train, y_test)
    
    # Compare results
    results = pd.DataFrame({
        "Model": ["Linear Regression", "Polynomial Regression", "Neural Network"],
        "MSE": [lr_mse, pr_mse, nn_mse],
        "MAE": [lr_mae, pr_mae, nn_mae],
        "R2": [lr_r2, pr_r2, nn_r2]
    })
    
    print("\nComparison of Results:")
    print(results)

if __name__ == "__main__":
    main()