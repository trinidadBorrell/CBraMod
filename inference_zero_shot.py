import torch
import torch.nn as nn
import numpy as np
import mne
from models.cbramod import CBraMod
from pathlib import Path
import json
from scipy.signal import resample
import os

'''
def create_256_to_64_roi_mapping():
    """Create mapping from 256-channel ROIs to 64-channel ROIs using the JSON mapping.
    
    Returns
    -------
    function
        Mapping function for ROI conversion
    """
    # Load the mapping file
    mapping_file = Path(__file__).parent / '..' / '..' / 'data' / 'egi256_biosemi64.json'
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Get the recombination groups (biosemi64 -> 256 electrodes)
    recombination_groups = mapping_data['recombination_groups']
    
    # Create reverse mapping: 256 electrode number -> 64 channel index
    electrode_256_to_ch_64 = {}
    
    # Define the actual channel order from the fif file
    ch_64_names_ordered = [
        'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
        'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
        'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
        'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
    ]
    
    # Convert electrode names to indices and create mapping
    for ch_64_name, electrode_256_list in recombination_groups.items():
        if ch_64_name in ch_64_names_ordered:
            ch_64_idx = ch_64_names_ordered.index(ch_64_name)  # 0-based indexing (0-63)
            
            # Convert electrode names like "E33" to electrode numbers
            for electrode_name in electrode_256_list:
                electrode_num = int(electrode_name[1:])  # E33 -> 33 (1-based EGI number)
                electrode_256_to_ch_64[electrode_num] = ch_64_idx
    
    def map_256_roi_to_64(roi_256):
        """Map a 256-channel ROI to corresponding 64-channel indices"""
        mapped_channels = set()
        for electrode_num in roi_256:
            if electrode_num in electrode_256_to_ch_64:
                mapped_channels.add(electrode_256_to_ch_64[electrode_num])
        return np.array(sorted(mapped_channels))
    
    return map_256_roi_to_64
'''

def preprocess_data(data_path, keep_256_channels=False):
    """
    Preprocess the input data for CBraMod inference.
    
    1. Loads the epochs from data from the specified path (already filtered)
    2. Segment the data into appropriate patches
    3. Normalize the data
    4. Convert to the expected tensor format: (batch_size, num_channels, time_segments, points_per_patch)
    
    Args:
        data_path (str): Path to the EEG fif file
        keep_256_channels (bool): If True, processes 256 channels without combining to 19
        
    Returns:
        tuple: (preprocessed_data, metadata_dict) where:
               - preprocessed_data: torch.Tensor with shape (batch_size, channels, segments, patch_size)
               - metadata_dict: contains usable_epochs, n_sequences, channel_names, sampling_rate, etc.
    """
    
    # Expected shape: (batch_size, num_channels, time_segments, points_per_patch)
    print(f"Preprocessing data from: {data_path}")

    # Load epochs from the specified path
    epochs = mne.read_epochs(data_path)
    print('Num electrodes: ', epochs.get_data().shape[1])
    
    # Get the raw data in shape (epochs, channels, time_points)
    raw_data = epochs.get_data()
    print(f'Raw data shape: {raw_data.shape}')  # (epochs, channels, time_points)
    
    if keep_256_channels == False:
        # Load mapping from 64 to 256 channels
        mapping_file = '/home/triniborrell/home/data/egi256_biosemi64.json'
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        
        electrode_64_to_256 = mapping_data['recombination_groups']
        pretrained_rois_64_electrode = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"] 
        
        # Check what channels are actually available in the data
        available_channels = epochs.ch_names
        
        # Combine multiple 256-channel electrodes per 64-channel electrode via mean
        combined_data = []
        missing_rois = []
        for roi in pretrained_rois_64_electrode:
            if roi in electrode_64_to_256:
                e256_electrodes = electrode_64_to_256[roi]
                # Find available electrodes for this ROI
                available_e256 = [ch for ch in e256_electrodes if ch in available_channels]
                if available_e256:
                    # Get indices and compute mean across channels (use whatever is available)
                    indices = [available_channels.index(ch) for ch in available_e256]
                    roi_data = raw_data[:, indices, :].mean(axis=1, keepdims=True)
                    combined_data.append(roi_data)
                    print(f"ROI {roi}: using {len(available_e256)}/{len(e256_electrodes)} electrodes: {available_e256}")
                else:
                    missing_rois.append(roi)
                    print(f"ROI {roi}: no electrodes available")
            else:
                missing_rois.append(roi)
                print(f"ROI {roi}: not in mapping")
        
        # Combine all ROIs and report missing ones
        if missing_rois:
            print(f"Warning: Completely missing electrodes for ROIs: {missing_rois}")
        print(f"Found {len(combined_data)}/{len(pretrained_rois_64_electrode)} electrodes")
        
        raw_data = np.concatenate(combined_data, axis=1)
        print(f'Combined data shape: {raw_data.shape}')
    
    print(f"Final data shape before processing: {raw_data.shape}")

    # Subsample the data --> From 250 Hz to 200 Hz
    time = raw_data.shape[-1]/250
    num_points_resampling = int(time*200)
    raw_data = resample(raw_data, num_points_resampling, axis=-1)
    print(f'Raw data shape after subsampling: {raw_data.shape}')
    
    # 2. Concatenate all epochs across time and segment into sequences
    patch_size = 200  # CBraMod expects patch_size=200 (matching in_dim=200)
    seq_len = 30      # CBraMod expects 30 consecutive patches per sample
    
    n_epochs, n_channels, n_timepoints = raw_data.shape
    epoch_duration_seconds = n_timepoints / 200
    print(f'Original data shape: {raw_data.shape}')
    print(f'Each epoch duration: {epoch_duration_seconds:.2f} seconds')
    
    # Store original number of epochs
    n_epochs_original = raw_data.shape[0]
    
    # Concatenate all epochs across time: (epochs, channels, time) -> (channels, total_time)
    concatenated_data = np.concatenate(raw_data, axis=1)  # Shape: (channels, total_time)
    print(f'Concatenated data shape: {concatenated_data.shape}')
    total_timepoints = concatenated_data.shape[1]
    print(f'Total duration after concatenation: {total_timepoints / 200:.2f} seconds')
    
    # Calculate how many complete 30-second sequences we can create
    total_patches = total_timepoints // patch_size
    n_sequences = total_patches // seq_len
    
    print(f'Total patches available: {total_patches}')
    print(f'Number of 30-second sequences: {n_sequences}')
    
    if n_sequences == 0:
        raise ValueError(f"Not enough data for even one 30-second sequence. Need at least {seq_len * patch_size} timepoints, have {total_timepoints}")
    
    # Trim data to fit complete sequences
    usable_timepoints = n_sequences * seq_len * patch_size
    trimmed_data = concatenated_data[:, :usable_timepoints]
    print(f'Trimmed data shape for complete sequences: {trimmed_data.shape}')
    
    # Reshape into sequences: (channels, n_sequences, seq_len, patch_size)
    sequences = trimmed_data.reshape(n_channels, n_sequences, seq_len, patch_size)
    
    # Transpose to get (n_sequences, channels, seq_len, patch_size)
    sequences = sequences.transpose(1, 0, 2, 3)
    print(f'Final sequences shape: {sequences.shape}')
    
    # Calculate actual usable epochs based on sequence time
    total_sequence_time_seconds = n_sequences * 30  # Each sequence is 30 seconds
    usable_epochs = int(total_sequence_time_seconds / epoch_duration_seconds)
    discarded_epochs = n_epochs_original - usable_epochs
    
    print(f'Usable epochs: {usable_epochs} (discarded {discarded_epochs} epochs)')
    print(f'Total usable time: {total_sequence_time_seconds:.1f} seconds')

    # Scale the sequences to the expected unit (100 uV)
    normalized_sequences = sequences * 1000 # --> the unit should be 100 uV
    tensor_data = torch.from_numpy(normalized_sequences).float()
    
    print(f"Final tensor shape: {tensor_data.shape}")  # (batch_size, channels, time_segments, patch_size)
    
    # Create metadata dictionary
    metadata = {
        'usable_epochs': usable_epochs,
        'total_epochs': n_epochs_original,
        'discarded_epochs': discarded_epochs,
        'n_sequences': n_sequences,
        'seq_len': seq_len,
        'patch_size': patch_size,
        'timepoints_per_epoch': int(epoch_duration_seconds * 200),  # Convert back to timepoints at 200Hz
        'epoch_duration_seconds': epoch_duration_seconds,
        'total_sequence_time_seconds': total_sequence_time_seconds,
        'sampling_rate': 200,  # After resampling
        'channel_names': epochs.ch_names if keep_256_channels else pretrained_rois_64_electrode,
        'n_channels': n_channels,
        'keep_256_channels': keep_256_channels
    }
    
    return tensor_data, metadata


def load_pretrained_model(model_path, device='cpu'):
    """
    Load the pretrained CBraMod model.
    
    Args:
        model_path (str): Path to the pretrained model weights
        device (str): Device to load the model on ('cpu' or 'cuda')
        
    Returns:
        CBraMod: Loaded model
    """
    # Initialize CBraMod with default parameters
    model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    
    # Load pretrained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded pretrained model from: {model_path}")
    except FileNotFoundError:
        print(f"Warning: Pretrained weights not found at {model_path}")
        print("Using randomly initialized model")
    
    # Move model to specified device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model


def inference(model, preprocessed_data, device='cpu'):
    """
    Perform inference using the CBraMod model.
    
    Args:
        model (CBraMod): Loaded CBraMod model
        preprocessed_data (torch.Tensor): Preprocessed input data
        device (str): Device to run inference on
        
    Returns:
        torch.Tensor: Model output (reconstructed signals)
    """
    # Move data to device
    preprocessed_data = preprocessed_data.to(device)
    
    # Perform forward pass
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(preprocessed_data)
    
    return output


def analyze_reconstruction(input_data, output_data):
    """
    Analyze reconstruction quality between input and output.
    
    Computes MSE, MAE, correlation, and signal statistics per patch and overall.
    
    Args:
        input_data (np.ndarray): Original input data (batch, channels, seq_len, patch_size)
        output_data (np.ndarray): Reconstructed output data (same shape)
    
    Returns:
        dict: Dictionary containing all computed metrics
    """
    print("\n" + "="*60)
    print("RECONSTRUCTION ANALYSIS")
    print("="*60)

    # Flatten for overall metrics
    input_flat = input_data.flatten()
    output_flat = output_data.flatten()
    
    # Overall metrics
    mse_overall = np.mean((input_data - output_data) ** 2)
    mae_overall = np.mean(np.abs(input_data - output_data))
    correlation_overall = np.corrcoef(input_flat, output_flat)[0, 1]
    
    # Check if sign flip improves correlation
    mse_flipped = np.mean((input_data - (-output_data)) ** 2)
    mae_flipped = np.mean(np.abs(input_data - (-output_data)))
    correlation_flipped = np.corrcoef(input_flat, -output_flat)[0, 1]

    
    print("\n--- OVERALL METRICS ---")
    print(f"MSE:         {mse_overall:.6f}")
    print(f"MAE:         {mae_overall:.6f}")
    print(f"Correlation: {correlation_overall:.6f}")
    print(f"MSE (sign-flipped output):         {mse_flipped:.6f}")
    print(f"MAE (sign-flipped output):         {mae_flipped:.6f}")
    print(f"Correlation (sign-flipped output): {correlation_flipped:.6f}")

    
    # Signal range statistics
    print("\n--- INPUT SIGNAL STATISTICS ---")
    print(f"Min:    {np.min(input_data):.4f}")
    print(f"Max:    {np.max(input_data):.4f}")
    print(f"Mean:   {np.mean(input_data):.4f}")
    print(f"Std:    {np.std(input_data):.4f}")
    print(f"Range:  {np.ptp(input_data):.4f}")
    
    print("\n--- OUTPUT SIGNAL STATISTICS ---")
    print(f"Min:    {np.min(output_data):.4f}")
    print(f"Max:    {np.max(output_data):.4f}")
    print(f"Mean:   {np.mean(output_data):.4f}")
    print(f"Std:    {np.std(output_data):.4f}")
    print(f"Range:  {np.ptp(output_data):.4f}")
    
    # Per-patch metrics
    for i in range(2):
        if i == 0: 
            output_data = output_data
        else:
            output_data = -output_data
        batch_size, n_channels, seq_len, patch_size = input_data.shape
        n_patches = batch_size * n_channels * seq_len
        
        input_patches = input_data.reshape(n_patches, patch_size)
        output_patches = output_data.reshape(n_patches, patch_size)
        
        mse_per_patch = np.mean((input_patches - output_patches) ** 2, axis=1)
        mae_per_patch = np.mean(np.abs(input_patches - output_patches), axis=1)
        
        # Correlation per patch
        corr_per_patch = np.array([
            np.corrcoef(input_patches[i], output_patches[i])[0, 1] 
            for i in range(n_patches)
        ])
        
        # Count patches with negative correlation (likely inverted)
        n_negative_corr = np.sum(corr_per_patch < 0)
        n_low_corr = np.sum(np.abs(corr_per_patch) < 0.5)
        
        print("\n--- PER-PATCH METRICS ---")
        print(f"Total patches: {n_patches}")
        print(f"MSE  - mean: {np.mean(mse_per_patch):.6f}, std: {np.std(mse_per_patch):.6f}, min: {np.min(mse_per_patch):.6f}, max: {np.max(mse_per_patch):.6f}")
        print(f"MAE  - mean: {np.mean(mae_per_patch):.6f}, std: {np.std(mae_per_patch):.6f}, min: {np.min(mae_per_patch):.6f}, max: {np.max(mae_per_patch):.6f}")
        print(f"Corr - mean: {np.mean(corr_per_patch):.6f}, std: {np.std(corr_per_patch):.6f}, min: {np.min(corr_per_patch):.6f}, max: {np.max(corr_per_patch):.6f}")
        print(f"Patches with negative correlation: {n_negative_corr} ({100*n_negative_corr/n_patches:.1f}%)")
        print(f"Patches with |corr| < 0.5:         {n_low_corr} ({100*n_low_corr/n_patches:.1f}%)")
        
        # Per-channel metrics
        print("\n--- PER-CHANNEL METRICS ---")
        for ch in range(n_channels):
            ch_input = input_data[:, ch, :, :].flatten()
            ch_output = output_data[:, ch, :, :].flatten()
            ch_mse = np.mean((ch_input - ch_output) ** 2)
            ch_corr = np.corrcoef(ch_input, ch_output)[0, 1]
            print(f"Channel {ch:2d}: MSE={ch_mse:.6f}, Corr={ch_corr:.6f}")
        
        print("="*60 + "\n")
        
    return {
        'mse_overall': mse_overall,
        'mae_overall': mae_overall,
        'correlation_overall': correlation_overall,
        'correlation_flipped': correlation_flipped,
        'mse_flipped': mse_flipped,
        'mae_flipped': mae_flipped}


def main():
    """
    Main function for zero-shot inference with CBraMod.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Zero-shot inference with CBraMod")
    parser.add_argument("--keep_256_channels", action="store_true",
                       help="Keep original 256-channel data for saving")
    parser.add_argument("--data_path", default="/data/project/eeg_foundation/data/test_data/fif_data_control_rs/sub-001/ses-01/sub-001_ses-01_task-rs_acq-01_epo_original.fif",
                       help="Path to input FIF file")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model
    model_path = '/home/triniborrell/home/projects/CBraMod/pretrained_weights/pretrained_weights.pth'
    model = load_pretrained_model(model_path, device)
    
    # Preprocess input data
    data_path = args.data_path
    preprocessed_data, metadata = preprocess_data(data_path, keep_256_channels=args.keep_256_channels)
    
    print(f"Model input data shape: {preprocessed_data.shape}")
    print(f"Metadata: {metadata}")
    
    # Perform inference
    output = inference(model, preprocessed_data, device)
    
    print(f"Output shape: {output.shape}")
    print("Inference completed successfully!")
    
    # Save results
    results_dir = '/home/triniborrell/home/projects/CBraMod/results_rs'
    os.makedirs(results_dir, exist_ok=True)

    # Save input and output data (normalized)
    input_np = preprocessed_data.cpu().numpy()
    np.save(os.path.join(results_dir, 'input_normalized.npy'), input_np)
    output_np = output.cpu().numpy()
    np.save(os.path.join(results_dir, 'model_output_normalized.npy'), output_np)
    
    # Save metadata
    np.save(os.path.join(results_dir, 'metadata.npy'), metadata)
    
    print(f"All results saved to {results_dir}")
    
    # Analyze reconstruction quality
    metrics = analyze_reconstruction(input_np, output_np)
    
    # Save metrics
    np.save(os.path.join(results_dir, 'reconstruction_metrics.npy'), metrics)
    print(f"Metrics saved to {results_dir}/reconstruction_metrics.npy")


if __name__ == "__main__":
    main()
