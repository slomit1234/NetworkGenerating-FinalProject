import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def minmax_scale(X, new_min, new_max):
    """Rescale time series to [new_min, new_max] range"""
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (new_max - new_min) + new_min
    return X_scaled

def gramian_angular_field(X, method='summation'):
    """Compute Gramian Angular Field (GAF)"""
    scaled_X = minmax_scale(X, -1, 1)  # rescale to [-1,1]
    phi = np.arccos(scaled_X)  # polar coordinate
    if method == 'summation':
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == 'difference':
        gaf = np.sin(phi[:, None] - phi[None, :])
    return gaf

# Define input and output directories
csv_dir = r'C:\Users\israe\NetDiffus\Youtube\Youtube\traces_80\actual'
output_dir = r'C:\Users\israe\NetDiffus\Youtube\Youtube\traces_80\output'

# Ensure the output directory exists
os.makedirs(csv_dir, exist_ok=True)

# Convert each CSV file to GAF
for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
        # Load the time-series data from the CSV
        csv_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(csv_path, header=None).values.flatten()
        # Convert to GAF
        gaf = gramian_angular_field(data, method='summation')

        # Plot and save as an image
        plt.imshow(gaf, cmap='rainbow')
        image_path = os.path.join(output_dir, filename.replace('.csv', '.png'))
        plt.savefig(image_path)
        plt.close()

        print(f"Converted {filename} to {image_path}")
