import os
import numpy as np
import pandas as pd
from PIL import Image

# Function to convert PNG to CSV format
def convert_image_to_csv(image_path, output_csv_dir, video_num, trace_num):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Convert the image to a NumPy array (shape will be [height, width, channels])
    img_array = np.array(img)
    
    # Check if the image has the right dimensions (128x128x3)
    if img_array.shape != (128, 128, 3):
        raise ValueError(f"Unexpected image shape: {img_array.shape}. Expected (128, 128, 3)")
    
    # Flatten the image into a 2D array with shape (128*128, 3)
    img_flattened = img_array.reshape(-1, 3)
    
    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_csv_dir, f"vid{video_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_file_path = os.path.join(output_dir, f"deepfp_vid{video_num}_{trace_num}.csv")
    pd.DataFrame(img_flattened).to_csv(csv_file_path, header=False, index=False)
    print(f"Saved: {csv_file_path}")

# Main function to process all images in the gasf_image directory
def process_images_to_csv(gasf_image_dir, output_csv_dir, num_of_videos):
    # Iterate over each video folder (assumed to be named as "vid1", "vid2", ..., "vid20")
    for v in range(1, num_of_videos + 1):
        video_dir = os.path.join(gasf_image_dir, f"vid{v}")
        
        if not os.path.exists(video_dir):
            print(f"Directory for video {v} not found: {video_dir}")
            continue
        
        # Get all PNG files in the current video directory
        png_files = [f for f in os.listdir(video_dir) if f.endswith('.png')]
        
        # Process each PNG file
        for i, png_file in enumerate(png_files):
            image_path = os.path.join(video_dir, png_file)
            convert_image_to_csv(image_path, output_csv_dir, v, i + 1)

if __name__ == "__main__":
    # Define the directory containing the GASF images
    gasf_image_dir = "/content/drive/MyDrive/Colab/Youtube_img/vid1"  # e.g., "/path/to/gasf_images/"
    
    # Define the directory where the CSV files will be saved
    output_csv_dir = "/content/drive/MyDrive/Colab/ActualForClassification"  # e.g., "/path/to/output_csv/"
    
    # Define the number of videos
    num_of_videos = 50  # Set this based on your actual data
    
    # Process the images and save them as CSV
    process_images_to_csv(gasf_image_dir, output_csv_dir, num_of_videos)
