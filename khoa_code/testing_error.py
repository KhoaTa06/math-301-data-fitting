from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def test_fitting_accuracy(data, predict_data, fitted_func):
    y_true = data['fxy'].values
    y_pred = np.array([fitted_func(x, y) for x, y in zip(data['x'], data['y'])])
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    predict_data['fxy'] = y_pred
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    return mse, r2


def calculate_slice_error(y_value, df, slice_fits):
    slice_df = df[df['y'] == y_value]
    if y_value not in slice_fits:
        print(f"No fitted function for y = {y_value}")
        return None, None
    
    x = slice_df['x'].values
    fxy_true = slice_df['fxy'].values
    A, B, C = slice_fits[y_value]
    fxy_pred = A * np.sin(B * x) + C
    
    mse = mean_squared_error(fxy_true, fxy_pred)
    r2 = r2_score(fxy_true, fxy_pred)
    
    print(f"Slice for y = {y_value}: MSE = {mse}, R2 = {r2}")
    return mse, r2