# Final-Project
# NetDiffus: Generating Synthetic Network Data for Security
# # Project Overview
This project implements the NetDiffus framework to generate synthetic network traffic using advanced machine learning models. The main goal is to create realistic synthetic datasets that support security testing without using sensitive real data, facilitating network security research and application.

# # Objectives
Generate synthetic network traffic: To create realistic network data for testing cybersecurity solutions.
Utilize diffusion models: Leverage diffusion models and data transformations for high-fidelity synthetic data.
Benchmark against existing datasets: Compare generated data to existing datasets to assess fidelity and reliability.

# # # Dataset Components
This project leverages three primary datasets:
D1: YouTube traffic data collected to represent multimedia streaming traffic.
D2: A publicly available Deep Fingerprinting dataset, widely used in traffic fingerprinting.
D3: A collection of network traffic data capturing diverse traffic conditions for added variety.

Key Components:
Dataset Preprocessing
Purpose: Convert raw traffic data to a format compatible with diffusion models.
Process: Includes feature extraction (e.g., packet directions and gaps), data normalization, and converting to 2D GASF images.
Normalization Challenges: During processing, different normalization methods were evaluated to match the original dataset’s properties.

# # # Diffusion Model
Model Choice: Utilizes diffusion models, which generate data by progressively adding and removing noise.
Image Transformation: Time-series data transformed into GASF images to leverage 2D model capabilities.
Training Setup: Training uses large GPU resources, often on platforms like Google Colab, and is fine-tuned with hyperparameter adjustments.

# # #Evaluation and Comparison
Metrics: The model's performance is evaluated using:
Visual inspection of generated data.
Classifier accuracy trained on synthetic data.
Unbiased FID scores to assess similarity to real data.

# #Project Setup
Prerequisites:
Python 3.x
Required libraries: numpy, pandas, pyts, cv2, matplotlib, scipy, skimage

# # Installation
Install the necessary libraries:

bash
Copy code
pip install numpy pandas pyts opencv-python-headless matplotlib scikit-image

## Running the Project
Dataset Preprocessing:
Place raw traffic files in DF/vid directory.
Run preprocess.py to generate GASF images and preprocessed datasets.
Training the Diffusion Model:
Use train_model.py to initiate model training on processed data.
Evaluation:
Use evaluate.py for calculating FID scores and performing classifier evaluations on generated samples.

## Acknowledgments
Special thanks to our instructors, Prof. Amit Dvir and Mr. Chen Hajaj, and to the authors of the Deep Fingerprinting and NetDiffus articles.
