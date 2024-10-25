import numpy as np
import matplotlib.pyplot as plt

# Load the saved samples
npz_file = "samples_16x128x128x3.npz"
data = np.load(npz_file)
print(data['arr_1'])
samples = data['arr_0']  # Assuming the generated images are stored under 'arr_0'


'''
for i in range(10): 
    plt.imshow(samples[i])
    plt.title(f"Sample {i+1}")
    plt.show()
'''