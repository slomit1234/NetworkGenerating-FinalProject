import numpy as np
import pandas as pd
import pickle
import cv2  # OpenCV for image resizing
from pyts.image import GramianAngularField
from pathlib import Path

def load_original_dataset(file_path):
    """
    Loads the original dataset from a pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')  # Specify the encoding
    return data

def extract_features(time_series_trace):
    """
    Extracts packet directions and inter-packet gaps from the raw traffic trace.
    """
    # Extract packet direction (+1 for outgoing, -1 for incoming based on sign)
    packet_directions = np.sign(time_series_trace)
    
    # Calculate inter-packet gaps
    inter_packet_gaps = np.diff(time_series_trace, prepend=0)
    
    return packet_directions, inter_packet_gaps

def pad_or_truncate(features, target_length=5000):
    """
    Ensures each trace has exactly 5000 samples by padding or truncating.
    """
    packet_directions, inter_packet_gaps = features
    
    # Truncate if longer than target_length
    if len(packet_directions) > target_length:
        packet_directions = packet_directions[:target_length]
        inter_packet_gaps = inter_packet_gaps[:target_length]
        
    # Pad with zeros if shorter than target_length
    else:
        pad_length = target_length - len(packet_directions)
        packet_directions = np.pad(packet_directions, (0, pad_length), 'constant')
        inter_packet_gaps = np.pad(inter_packet_gaps, (0, pad_length), 'constant')
        
    return packet_directions, inter_packet_gaps

def convert_to_gasf_image(time_series_trace, image_size=5000):
    """
    Converts the 1D time-series trace into a 2D GASF image.
    """
    # Normalize the trace to the range [-1, 1] as mentioned in the article for better contrast.
    min_val, max_val = np.min(time_series_trace), np.max(time_series_trace)
    normalized_trace = (2 * (time_series_trace - min_val) / (max_val - min_val)) - 1
    
    # Convert to GASF image using pyts
    gasf = GramianAngularField(image_size=image_size, method='summation')
    gasf_image = gasf.fit_transform(normalized_trace.reshape(1, -1))[0]
    
    return gasf_image

def apply_gamma_correction(image, gamma=0.25):
    """
    Applies gamma correction to the GASF image, ensuring all values are valid.
    """
    # Replace NaN values with 0
    image = np.nan_to_num(image, nan=0.0)
    
    # Ensure all values are non-negative
    image[image < 0] = 0
    
    # Apply gamma correction: I_c = A * I_r^γ where A = 1
    corrected_image = np.power(image, gamma)
    
    return corrected_image

def resize_image(image, size=(125, 125)):
    """
    Resizes the GASF image to the target size using INTER_AREA interpolation.
    """
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

def normalize_image(image):
    """
    Normalizes pixel values to the range [0, 1].
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def process_trace(trace, output_dir, trace_id):
    """
    Full preprocessing pipeline for a single trace, saves each trace as a separate CSV file.
    """
    # Extract features
    features = extract_features(trace)
    
    # Pad or truncate to 5000 samples
    padded_features = pad_or_truncate(features)
    
    # Combine packet direction and inter-packet gaps into a single trace
    combined_trace = np.array(padded_features[0]) + np.array(padded_features[1])
    
    # Convert to GASF image
    gasf_image = convert_to_gasf_image(combined_trace)
    
    # Apply gamma correction
    gamma_corrected_image = apply_gamma_correction(gasf_image)
    
    # Resize image
    resized_image = resize_image(gamma_corrected_image)
    
    # Normalize the image
    normalized_image = normalize_image(resized_image)
    
    # Save as a separate CSV file
    output_file = output_dir / f'trace_{trace_id}.csv'
    pd.DataFrame(normalized_image).to_csv(output_file, index=False, header=False)
    
    print(f"Trace {trace_id} saved to {output_file}")

def main(input_path, output_dir):
    """
    Main function that processes all traces and saves them as separate CSV files.
    """
    # Load original data
    X_valid_NoDef = load_original_dataset(input_path / 'X_valid_NoDef.pkl')
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each trace
    for trace_id, trace in enumerate(X_valid_NoDef):
        process_trace(trace, output_dir, trace_id)

if __name__ == "__main__":
    # Define paths to input and output data
    input_data_path = Path('C:/Users/stava/Deep/input')  # Adjust the path as needed
    output_dir = Path('C:/Users/stava/Deep/output')  # Directory where each CSV trace will be saved
    
    main(input_data_path, output_dir)
