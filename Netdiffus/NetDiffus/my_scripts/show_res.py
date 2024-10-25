import os
import numpy as np
import matplotlib.pyplot as plt

# Load the .npz file
data = np.load('samples_512x128x128x3.npz')

output_dir = r'outputimages'
os.makedirs(output_dir, exist_ok=True)

# Extract the images and labels
images = data['arr_0']  # assuming 'arr_0' contains images
labels = data['arr_1']  # assuming 'arr_1' contains labels

# Display a few images
for i in range(512):  # Display first 5 images
    img = images[i]
    img = img.astype(np.uint8)  # Convert to uint8 for visualization
    #plt.imshow(img)
    plt.title(f"Sample {i+1}, Label: {labels[i]}")
    #plt.show()
    plt.imsave(f'{output_dir}/generated_{i}.png', img)
