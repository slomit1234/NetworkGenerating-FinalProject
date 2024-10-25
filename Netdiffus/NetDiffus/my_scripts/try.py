import numpy as np
import os
import pandas as pd

# Load the synthetic data from the npz file
npz_file = '/content/drive/MyDrive/Colab/synth_models/samples.npz'
output_dir = 'synth_try'

# Load the npz data
data = np.load(npz_file)
all_samples = data['arr_0']  # Assuming arr_0 contains the samples
all_labels = data['arr_1']   # Assuming arr_1 contains the labels for vid1 and vid2 classification

# Create the output directories for vid1 and vid2
vid1_dir = os.path.join(output_dir, 'vid1')
vid2_dir = os.path.join(output_dir, 'vid2')
os.makedirs(vid1_dir, exist_ok=True)
os.makedirs(vid2_dir, exist_ok=True)

# Save samples into the appropriate directories
for idx, (sample, label) in enumerate(zip(all_samples, all_labels)):
    sample_flattened = sample.reshape(-1, sample.shape[-1])  # Flatten the sample
    df = pd.DataFrame(sample_flattened)
    
    if label == 0:  # Assuming label 0 corresponds to vid1
        file_path = os.path.join(vid1_dir, f'{idx + 1}.csv')
    else:  # Assuming label 1 corresponds to vid2
        file_path = os.path.join(vid2_dir, f'{idx + 1}.csv')
    
    # Save each sample as a CSV
    df.to_csv(file_path, index=False, header=False)

print("Synthetic data saved successfully!")
