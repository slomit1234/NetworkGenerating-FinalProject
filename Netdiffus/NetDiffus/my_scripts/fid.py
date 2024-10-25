import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import os

# Function to calculate FID score between real and generated features
def calculate_fid(real_features, fake_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to extract features using the InceptionV3 model
def extract_inception_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            preds = model(images)
            features.append(preds.cpu().numpy())
    return np.concatenate(features, axis=0)

# Load your test dataset
def load_test_data(test_path):
    # Load your dataset here
    # This is a placeholder function; you need to load your CSV images data accordingly
    # Return DataLoader
    pass

# Evaluation function
def evaluate(test_loader, model, criterion, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    losses = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    return np.array(true_labels), np.array(predicted_labels), np.mean(losses)

# Main function for evaluation
def evaluate_metrics(test_loader, real_loader, fake_loader, model, device, criterion):
    # Calculate basic metrics
    true_labels, predicted_labels, test_loss = evaluate(test_loader, model, criterion, device)
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    confusion = confusion_matrix(true_labels, predicted_labels)

    print(f"Test Loss: {test_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")

    # Load pretrained InceptionV3 model for FID calculation
    inception = inception_v3(pretrained=True, transform_input=False)
    inception.fc = torch.nn.Identity()
    inception.to(device)

    # Extract features from real and fake datasets using InceptionV3
    real_features = extract_inception_features(real_loader, inception, device)
    fake_features = extract_inception_features(fake_loader, inception, device)

    # Calculate FID score
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define your model, test dataset, and criterion
    model = DNN_model(num_channels=3, classes=20)  # Your custom DNN model
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Load the test dataset
    test_loader = load_test_data("/path/to/test/data")
    real_loader = load_test_data("/path/to/real/data")
    fake_loader = load_test_data("/path/to/fake/data")

    # Run evaluation
    evaluate_metrics(test_loader, real_loader, fake_loader, model, device, criterion)
