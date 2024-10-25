import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Confusion matrix data from the image
confusion_matrix = np.array([[0, 32],
                             [0, 32]])

# Create a heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)

# Add labels
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Display the heatmap
plt.show()
