import os
import numpy as np
import pandas as pd

# Function to check the shape of the CSV file and verify if it suits the classifier
def analyze_csv_file(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file, header=None).values
    
    # Check the shape of the data
    expected_shape = (128 * 128, 3)  # We expect a 2D array with shape (128*128, 3)
    
    if data.shape == expected_shape:
        print(f"{csv_file}: Shape is OK ({data.shape})")
    else:
        print(f"{csv_file}: Unexpected shape {data.shape}. Expected {expected_shape}.")
    
    # Attempt to reshape into [3, 128, 128] as required by the classifier
    try:
        reshaped_data = data.reshape(3, 128, 128)
        print(f"{csv_file}: Successfully reshaped to {reshaped_data.shape} (3, 128, 128)")
    except ValueError as e:
        print(f"{csv_file}: Failed to reshape to (3, 128, 128) - {str(e)}")

# Main function to analyze all CSV files in the directory
def analyze_all_csv_files(actual_csv_dir):
    # Walk through all the CSV files in the directory
    for root, dirs, files in os.walk(actual_csv_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_file_path = os.path.join(root, file)
                analyze_csv_file(csv_file_path)

if __name__ == "__main__":
    # Path to the directory containing the CSV files
    actual_csv_dir = "/content/drive/MyDrive/Colab/ActualForClassification"   # Replace with your actual path

    # Analyze all CSV files
    analyze_all_csv_files(actual_csv_dir)
