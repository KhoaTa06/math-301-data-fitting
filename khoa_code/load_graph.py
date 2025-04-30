import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def interactive_3d_plot(dataset, predict_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=dataset['x'], y=dataset['y'], z=dataset['fxy'], mode='markers', marker=dict(size=2, color='blue', colorscale='Viridis'), name='Data Points'))
    
    fig.add_trace(go.Scatter3d(x=predict_data['x'], y=predict_data['y'], z=predict_data['fxy'], mode='markers', marker=dict(size=2, color='red', colorscale='Viridis'), name='Predicted Points'))


    fig.update_layout(title='Interactive 3D Plot of Data Points',
                      scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='f(X, Y)'))
    fig.show()
    

def plot_y_slice(dataset, predicted_data, figure_num=3):
    # Validate inputs
    if not isinstance(dataset, pd.DataFrame) or not all(col in dataset.columns for col in ['x', 'y', 'fxy']):
        print("Error: Dataset must be a Pandas DataFrame with 'x', 'y', 'fxy' columns.")
        return
    if not isinstance(predicted_data, pd.DataFrame) or not all(col in predicted_data.columns for col in ['x', 'y', 'fxy']):
        print("Error: Predicted_data must be a Pandas DataFrame with 'x', 'y', 'fxy' columns.")
        return
    
    # Get unique y-values (assuming they are the same in both datasets)
    y_values = sorted(dataset['y'].unique())
    if not y_values:
        print("No unique y-values found.")
        return
    
    plt.figure(figure_num, figsize=(8, 6))  # Unique figure number
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.2)
    
    class PlotNavigator:
        def __init__(self):
            self.index = 0
            self.plot_current_y()
        
        def plot_current_y(self):
            ax.clear()
            y_val = y_values[self.index]
            
            # Plot actual data
            filtered_data = dataset[dataset['y'] == y_val]
            x_values = filtered_data['x'].values
            fxy_values = filtered_data['fxy'].values
            sort_indices = x_values.argsort()
            x_values = x_values[sort_indices]
            fxy_values = fxy_values[sort_indices]
            ax.plot(x_values, fxy_values, 'r-', label=f'Actual (y = {y_val})')
            
            # Plot predicted data
            filtered_pred = predicted_data[predicted_data['y'] == y_val]
            x_pred = filtered_pred['x'].values
            fxy_pred = filtered_pred['fxy'].values
            sort_indices_pred = x_pred.argsort()
            x_pred = x_pred[sort_indices_pred]
            fxy_pred = fxy_pred[sort_indices_pred]
            ax.plot(x_pred, fxy_pred, 'b--', label=f'Predicted (y = {y_val})')
            
            ax.set_ylim(-8, 8)
            ax.set_xlabel('x')
            ax.set_ylabel('fxy')
            ax.set_title(f'Actual vs Predicted fxy for y = {y_val}')
            ax.legend()
            ax.grid(True)
            plt.draw()
        
        def prev(self, event):
            self.index = (self.index - 1) % len(y_values)
            self.plot_current_y()
        
        def next(self, event):
            self.index = (self.index + 1) % len(y_values)
            self.plot_current_y()

    navigator = PlotNavigator()
    ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    btn_prev.on_clicked(navigator.prev)
    btn_next.on_clicked(navigator.next)
    plt.show()


def plot_x_slice(dataset, predicted_data, figure_num=2):
    # Validate inputs
    if not isinstance(dataset, pd.DataFrame) or not all(col in dataset.columns for col in ['x', 'y', 'fxy']):
        print("Error: Dataset must be a Pandas DataFrame with 'x', 'y', 'fxy' columns.")
        return
    if not isinstance(predicted_data, pd.DataFrame) or not all(col in predicted_data.columns for col in ['x', 'y', 'fxy']):
        print("Error: Predicted_data must be a Pandas DataFrame with 'x', 'y', 'fxy' columns.")
        return
    
    # Get unique x-values (assuming they are the same in both datasets)
    x_values = sorted(dataset['x'].unique())
    if not x_values:
        print("No unique x-values found.")
        return
    
    plt.figure(figure_num, figsize=(8, 6))  # Unique figure number
    ax = plt.gca()
    plt.subplots_adjust(bottom=0.2)
    
    class PlotNavigator:
        def __init__(self):
            self.index = 0
            self.plot_current_x()
        
        def plot_current_x(self):
            ax.clear()
            x_val = x_values[self.index]
            
            # Plot actual data
            filtered_data = dataset[dataset['x'] == x_val]
            y_values = filtered_data['y'].values
            fxy_values = filtered_data['fxy'].values
            sort_indices = y_values.argsort()
            y_values = y_values[sort_indices]
            fxy_values = fxy_values[sort_indices]
            ax.plot(y_values, fxy_values, 'r-', label=f'Actual (x = {x_val})')
            
            # Plot predicted data
            filtered_pred = predicted_data[predicted_data['x'] == x_val]
            y_pred = filtered_pred['y'].values
            fxy_pred = filtered_pred['fxy'].values
            sort_indices_pred = y_pred.argsort()
            y_pred = y_pred[sort_indices_pred]
            fxy_pred = fxy_pred[sort_indices_pred]
            ax.plot(y_pred, fxy_pred, 'b--', label=f'Predicted (x = {x_val})')
            
            ax.set_ylim(-8, 8)
            ax.set_xlabel('y')
            ax.set_ylabel('fxy')
            ax.set_title(f'Actual vs Predicted fxy for x = {x_val}')
            ax.legend()
            ax.grid(True)
            plt.draw()
        
        def prev(self, event):
            self.index = (self.index - 1) % len(x_values)
            self.plot_current_x()
        
        def next(self, event):
            self.index = (self.index + 1) % len(x_values)
            self.plot_current_x()

    navigator = PlotNavigator()
    ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    btn_prev.on_clicked(navigator.prev)
    btn_next.on_clicked(navigator.next)
    plt.show()