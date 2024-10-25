import os
import numpy as np
import pandas as pd

# Load the npz file
data = np.load("samples_512x128x128x3.npz")

# Extract the images and labels
arr_images = data['arr_0']  # The images
arr_labels = data['arr_1']  # The labels associated with the images

# Define the output directory
output_dir = "synth"
os.makedirs(output_dir, exist_ok=True)

# Loop through the unique labels (videos)
for label in np.unique(arr_labels):
    # Create a directory for each video
    video_dir = os.path.join(output_dir, f"vid{label+1}")
    os.makedirs(video_dir, exist_ok=True)
    
    # Get all the indices for this label
    indices = np.where(arr_labels == label)[0]
    
    # Loop through each sample for this video
    for i, idx in enumerate(indices):
        # Extract the corresponding image
        image = arr_images[idx]
        
        # Save each channel (3 channels) of the image in its original form as a CSV file
        output_file = os.path.join(video_dir, f"{i+1}.csv")
        
        # Convert the image to 2D and save it
        # Since each image is 128x128x3, we store it as rows = 128, columns = 3*128 (channel interleaved)
        reshaped_image = image.reshape(image.shape[0], -1)
        pd.DataFrame(reshaped_image).to_csv(output_file, header=False, index=False)
        
        print(f"Saved {output_file}")
