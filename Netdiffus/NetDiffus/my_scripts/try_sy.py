import os
import numpy as np
import cv2
import pandas as pd

# Function to normalize GASF images to [0,1] range
def normalize_gasf(gasf_image):
    return gasf_image / 255.0

# Function to apply gamma correction to enhance contrast
def apply_gamma_correction(gasf_image, gamma=0.25):
    inv_gamma = 1.0 / gamma
    corrected = np.power(gasf_image, inv_gamma)
    return corrected

# Function to resize the GASF images to the target shape (125, 125)
def resize_gasf(gasf_image, target_shape=(125, 125)):
    resized_image = cv2.resize(gasf_image, target_shape, interpolation=cv2.INTER_AREA)
    return resized_image

# Function to save the GASF images as CSV files in the desired structure
def save_csv_files(gasf_data, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directories for vid1 and vid2
    vid1_dir = os.path.join(output_dir, 'vid1')
    vid2_dir = os.path.join(output_dir, 'vid2')
    os.makedirs(vid1_dir, exist_ok=True)
    os.makedirs(vid2_dir, exist_ok=True)
    
    for i, (gasf_image, label) in enumerate(zip(gasf_data, labels)):
        # Normalize, apply gamma correction, and resize the image
        gasf_normalized = normalize_gasf(gasf_image)
        gasf_corrected = apply_gamma_correction(gasf_normalized)
        gasf_resized = resize_gasf(gasf_corrected)
        
        # Extract just one channel (e.g., the first channel) to save as 2D CSV
        gasf_resized_single_channel = gasf_resized[:, :, 0]
        
        # Save the image in the corresponding vid directory
        if label == 0:
            csv_path = os.path.join(vid1_dir, f'{i + 1}.csv')
        else:
            csv_path = os.path.join(vid2_dir, f'{i + 1}.csv')
        
        # Save as CSV (2D format)
        pd.DataFrame(gasf_resized_single_channel).to_csv(csv_path, header=False, index=False)

# Main function to load the npz file and save CSV files in the required structure
def main(npz_file, output_dir):
    # Load the npz file
    data = np.load(npz_file)
    gasf_data = data['arr_0']  # Assuming the GASF images are stored in 'arr_0'
    labels = data['arr_1']     # Assuming the labels are stored in 'arr_1'
    
    # Save the GASF images as CSV files
    save_csv_files(gasf_data, labels, output_dir)

if __name__ == '__main__':
    npz_file = 'samples_512x128x128x3.npz'  # Your npz file path
    output_dir = 'synth2'  # Directory to save the CSV files
    main(npz_file, output_dir)
