import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define input directory paths
gan_dir = r'gan'
#synth_dir = 'Youtube/synth'

output_dir = r'ganimages'
os.makedirs(output_dir, exist_ok=True)

# Function to load samples from directories
def load_samples_from_csv(directory):
    images = []
    labels = []

    # Traverse all video folders inside the directory (e.g., vid1, vid2, ...)
    for vid_folder in os.listdir(directory):
        vid_path = os.path.join(directory, vid_folder)
        #print(vid_path)
        if os.path.isdir(vid_path):
            for sample_file in os.listdir(vid_path):
                #print(sample_file)
                if sample_file.endswith('.csv'):
                    # Load CSV file as numpy array
                    csv_path = os.path.join(vid_path, sample_file)
                    sample_data = pd.read_csv(csv_path, header=None).to_numpy()
                    #print(sample_data.size)
                    # Assuming each CSV represents a 128x128x3 image flattened (128 * 128 * 3 = 49152)
                    if sample_data.size == 15625:
                        sample_image = sample_data.reshape((125, 125))
                        images.append(sample_image)

                        # Extract label from folder name, assuming folder name indicates the video label (e.g., vid1 -> label 1)
                        label = int(vid_folder.replace('vid', ''))
                        labels.append(label)
    
    #print(images)
    return np.array(images), np.array(labels)

# Load images and labels from both gan and synth directories
gan_images, gan_labels = load_samples_from_csv(gan_dir)
#synth_images, synth_labels = load_samples_from_csv(synth_dir)

# Combine gan and synth data if needed
#all_images = np.concatenate((gan_images, synth_images), axis=0)
#all_labels = np.concatenate((gan_labels, synth_labels), axis=0)
num = len(gan_images)
print(num)
# Save a few images for visualization
for i in range(num):
    img = gan_images[i].astype(np.uint8)  # Convert to uint8 for visualization
    plt.title(f"Sample {i+1}, Label: {gan_labels[i]}")
    plt.imsave(f'{output_dir}/generated_{i}.png', img)

print(f"Processed {len(gan_images)} images and saved them to {output_dir}.")
