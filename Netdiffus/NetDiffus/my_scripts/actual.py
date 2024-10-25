import pandas as pd
import numpy as np
import os
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from skimage.transform import resize



def process_and_save_data(input_dir, output_dir):
    """
    Process all actual data files, apply GASF, rescale, gamma correction,
    and save the processed data in the desired structure.
    """
    num_of_vid = 2  # Assuming we have vid1 and vid2

    for v in range(1, num_of_vid + 1):
        video_dir = os.path.join(input_dir, f"vid{v}")
        csv_files = [f for f in os.listdir(video_dir) if f.endswith('.csv')]

        for i, csv_file in enumerate(csv_files):
            df = pd.read_csv(file, usecols=[' addr2_bytes'])  # the column to read the data
            df = df.replace(np.nan, 0)
            data = df.to_numpy(dtype='float')
            pts = []
            # GASF formation
            for l in data:
                for j in l:
                    pts.append(j)
            num_of_samples_per_bin = 4  # to bin the video data
            points = []
            for j in range(int(len(pts) / num_of_samples_per_bin)):
                points.append(np.sum(pts[j * num_of_samples_per_bin:(j + 1) * num_of_samples_per_bin]))
            X = np.array([points])
            # Compute Gramian angular fields
            gasf = GramianAngularField(sample_range=(0, 1), method='summation')
            X_gasf = gasf.fit_transform(X)
            # Gamma correction
            gasf_img = X_gasf[0] * 0.5 + 0.5
            gamma = 0.25
            gasf_img = np.power(gasf_img, gamma)
            # Save processed data
            save_processed_data(gamma_corrected_data, v, i + 1, output_dir)
            

if __name__ == "__main__":
    input_dir = "actual"  # Path to the actual data
    output_dir = "processed_actual"    # Directory to save the processed data
    process_and_save_data(input_dir, output_dir)
