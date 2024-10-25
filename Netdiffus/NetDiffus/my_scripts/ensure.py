import pandas as pd
import numpy as np
from skimage.transform import resize
import os

# Load and check the data shape
def load_and_check_data(csv_file):
    # Load the data from CSV
    data = pd.read_csv(csv_file, header=None).values
    
    print(f"Checking file: {csv_file}")
    print(f"Original data shape: {data.shape}")
    
    # Check if the shape is 2D (which is expected before converting to 3D)
    if len(data.shape) == 2:
        height, width = data.shape
        print(f"2D data detected with shape: {height}x{width}")
        
        # Assume the data should be resized to 128x128 if it's not already
        if height != 128 or width != 128:
            print(f"Resizing data to 128x128")
            data_resized = resize(data, (128, 128), anti_aliasing=True)
        else:
            data_resized = data
        
        # Convert to 3-channel (RGB-like) by stacking the same data 3 times
        data_3ch = np.stack([data_resized] * 3, axis=0)
        print(f"Resized and stacked data shape: {data_3ch.shape}")
        
    elif len(data.shape) == 3:
        print("Data is already 3D, checking if shape matches model input")
        data_3ch = data
        if data_3ch.shape != (3, 128, 128):
            print(f"Resizing 3D data to match expected shape (3, 128, 128)")
            data_3ch_resized = resize(data_3ch, (3, 128, 128), anti_aliasing=True)
            data_3ch = data_3ch_resized
            print(f"Resized data shape: {data_3ch.shape}")
        else:
            print("Data shape matches expected shape.")
        
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    return data_3ch

# Save the rescaled data
def save_rescaled_data(data, output_dir, file_name):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    
    # Flatten the data to 2D (for saving as CSV) with shape (128*128, 3)
    data_flattened = data.reshape(128 * 128, 3)
    
    # Save as CSV
    pd.DataFrame(data_flattened).to_csv(output_path, header=False, index=False)
    print(f"Data saved to: {output_path}")

# Main function to run the process for every file in the directory
def process_directory(input_dir, output_dir):
    """
    Process all files in the directory, check and resize if needed, then save.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                try:
                    # Load and check the data
                    data_3ch = load_and_check_data(csv_file)

                    # Save the properly formatted data
                    file_name = os.path.basename(csv_file)
                    save_rescaled_data(data_3ch, output_dir, file_name)

                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")

if __name__ == "__main__":
    input_dir = "C:\Users\israe\Downloads\27_09\all"  # Path to the directory with all CSV files
    output_dir = "processed_synth"  # Directory to save the processed data

    process_directory(input_dir, output_dir)
