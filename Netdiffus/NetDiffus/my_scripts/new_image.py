import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

def normalize_data(X):
    """Normalize time series data to [-1, 1] range"""
    return (X - X.min()) / (X.max() - X.min()) * 2 - 1

def polar_transform(X):
    """Convert normalized data to polar representation"""
    return np.arccos(X)

def gramian_angular_field(X, method='summation'):
    """Compute Gramian Angular Field (GAF)"""
    phi = polar_transform(normalize_data(X))  # Convert to polar coordinates
    if method == 'summation':
        gaf = np.cos(phi[:, None] + phi[None, :])
    elif method == 'difference':
        gaf = np.sin(phi[:, None] - phi[None, :])
    return gaf

def apply_gamma_correction(image, gamma=0.25):
    """Apply gamma correction to enhance contrast"""
    inv_gamma = 1.0 / gamma
    corrected = np.power(image, inv_gamma)
    return corrected

# Directories

csv_dir = r'C:\Users\israe\NetDiffus\DF\DF\traces_80\actual'
output_dir = r'C:\Users\israe\NetDiffus\DF\DF\traces_80\output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert each CSV file to GAF images and apply gamma correction
for filename in os.listdir(csv_dir):
    if filename.endswith('.csv'):
        # Load the time-series data from the CSV
        csv_path = os.path.join(csv_dir, filename)
        data = pd.read_csv(csv_path, header=None).values.flatten()
        
        # Create GASF and GADF images
        gasf = gramian_angular_field(data, method='summation')
        gadf = gramian_angular_field(data, method='difference')

        # Apply gamma correction to GASF
        gasf_corrected = apply_gamma_correction(gasf)

        # Resize GASF using OpenCV
        resized_gasf = cv2.resize(gasf_corrected, (128, 128), interpolation=cv2.INTER_AREA)

        # Save the images
        plt.imsave(os.path.join(output_dir, f"{filename.replace('.csv', '_GASF.png')}"), resized_gasf, cmap='rainbow')
        plt.imsave(os.path.join(output_dir, f"{filename.replace('.csv', '_GADF.png')}"), gadf, cmap='rainbow')

        print(f"Converted {filename} to GASF and GADF with resizing and gamma correction.")
