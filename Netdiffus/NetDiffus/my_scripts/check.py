import pandas as pd
author_gasf = pd.read_csv('Youtube_1.csv', header=None)
user_gasf = pd.read_csv('deepfp_vid1_1.csv', header=None)

author_gasf.head()
user_gasf.head()

print("Author GASF Shape:", author_gasf.shape)
print("User GASF Shape:", user_gasf.shape)

print("Author GASF Stats:", author_gasf.describe())
print("User GASF Stats:", user_gasf.describe())

difference = (author_gasf - user_gasf).abs()
print("Max difference:", difference.max().max())
