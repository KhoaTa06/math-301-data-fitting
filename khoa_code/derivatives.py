import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline

def compute_derivatives(dataset, smoothing_factor=0.5):
    """
    Compute partial derivatives df/dx and df/dy of f(x,y) data using spline smoothing.
    
    Parameters:
    - dataset: Pandas DataFrame with columns 'x', 'y', 'fxy'
    - smoothing_factor: Smoothing factor for UnivariateSpline (default: 0.5)
    
    Returns:
    - DataFrame with added columns 'df_dx' and 'df_dy'
    """
    # Validate input
    if not isinstance(dataset, pd.DataFrame) or not all(col in dataset.columns for col in ['x', 'y', 'fxy']):
        raise ValueError("Dataset must be a Pandas DataFrame with 'x', 'y', 'fxy' columns.")
    
    # Create a copy of the DataFrame
    data = dataset.copy()
    
    # Initialize columns for derivatives
    data['df_dx'] = np.nan
    data['df_dy'] = np.nan
    
    # Compute derivative w.r.t. y for each fixed x (x-slice)
    x_values = sorted(data['x'].unique())
    for x_val in x_values:
        subset = data[data['x'] == x_val].sort_values('y')
        y_vals = subset['y'].values
        fxy_vals = subset['fxy'].values
        # Smooth with spline
        spline = UnivariateSpline(y_vals, fxy_vals, s=smoothing_factor)
        # Compute derivative w.r.t. y
        df_dy = spline.derivative()(y_vals)
        data.loc[data['x'] == x_val, 'df_dy'] = df_dy
    
    # Compute derivative w.r.t. x for each fixed y (y-slice)
    y_values = sorted(data['y'].unique())
    for y_val in y_values:
        subset = data[data['y'] == y_val].sort_values('x')
        x_vals = subset['x'].values
        fxy_vals = subset['fxy'].values
        # Smooth with spline
        spline = UnivariateSpline(x_vals, fxy_vals, s=smoothing_factor)
        # Compute derivative w.r.t. x
        df_dx = spline.derivative()(x_vals)
        data.loc[data['y'] == y_val, 'df_dx'] = df_dx
    
    return data