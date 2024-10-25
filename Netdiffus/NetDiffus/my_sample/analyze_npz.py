import numpy as np

# Load the npz file
data = np.load("samples_512x128x128x3.npz")

# Extract the arrays
arr_images = data['arr_0']  # The images
arr_labels = data['arr_1']  # The labels associated with the images

# Print the shape of the images and labels
print("Shape of images array:", arr_images.shape)  # Should be (num_samples, height, width, channels)
print("Shape of labels array:", arr_labels.shape)  # Should be (num_samples,)

# Unique labels (videos) and count of samples per label (video)
unique_labels, counts = np.unique(arr_labels, return_counts=True)
print("Unique video labels (videos):", unique_labels)
print("Number of samples per video:", counts)
