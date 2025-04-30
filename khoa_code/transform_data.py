import pandas as pd
import numpy as np
import math
from load_graph import interactive_3d_plot, plot_y_slice, plot_x_slice

dataFilepath = "/Users/khoa/Desktop/Math_301/math-301-data-fitting/khoa_code/Group_1_Data_10000.txt"
data = pd.read_csv(dataFilepath)
data.columns
data.drop('y)', axis=1, inplace=True)
data.columns=["x", "y", "fxy"]

def tangent(x):
    return np.tan(x)

sine_data = data.copy()
sine_data['fxy'] = (sine_data['fxy'])*(np.sin(sine_data['x']))
sine_data.to_csv("tan_data.csv", index=False)

interactive_3d_plot(sine_data)