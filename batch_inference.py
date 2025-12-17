#!/usr/bin/env python3
"""
Batch inference script for CBraMod.

Processes multiple EEG recordings through CBraMod and saves reconstructed data as FIF files.

Usage:
    python batch_inference.py --use_19_electrodes --subjects all
    python batch_inference.py --use_256_electrodes --subjects sub-001 sub-002
"""

import os
import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import mne
import json
from scipy.signal import resample

from models.cbramod import CBraMod
from inference_zero_shot import preprocess_data, load_pretrained_model, inference


def find_all_fif_files(base_dir):
    """
    Find all FIF files in the directory structure.
    
    Args:
        base_dir (str): Base directory containing subject folders
        
    Returns:
        list: List of tuples (subject_id, session_num, file_path)
    """
    fif_files = []
    base_path = Path(base_dir)
    
    # Find all subjects
    for subject_dir in base_path.glob("sub-*"):
        subject_id = subject_dir.name
        
        # Find all sessions
        for session_dir in subject_dir.glob("ses-*"):
            session_num = session_dir.name.split("-")[1]
            
            # Find FIF files
            for fif_file in session_dir.glob("eeg/sub-*_ses-*_task-rs_acq-01_epo.fif"):
                fif_files.append((subject_id, session_num, str(fif_file)))
    
    return sorted(fif_files)


def reconstruct_to_fif(model_output, original_epochs, metadata):
    """
    Convert model output back to FIF format.
    
    Args:
        model_output (torch.Tensor): Model output (batch, channels, seq_len, patch_size)
        original_epochs (mne.Epochs): Original epochs object for metadata
        metadata (dict): Metadata from preprocess_data containing usable_epochs, channel_names, etc.
        
    Returns:
        mne.Epochs: Reconstructed epochs with original metadata
    """
    # Convert to numpy and scale back to Volts
    output_np = model_output.cpu().numpy()
    output_np = output_np / 1000  # Convert back from 100µV to Volts
    
    # Reshape from (n_sequences, channels, seq_len, patch_size) to (n_epochs, channels, timepoints)
    batch_size, n_channels, seq_len, patch_size = output_np.shape
    n_sequences = batch_size
    total_timepoints = batch_size * seq_len * patch_size
    
    # Concatenate all sequences
    reconstructed_data = output_np.transpose(0, 2, 1, 3)  # (batch, seq_len, channels, patch_size)
    reconstructed_data = reconstructed_data.reshape(total_timepoints, n_channels)
    
    # Use metadata to determine the correct number of epochs and timepoints
    usable_epochs = metadata['usable_epochs']
    timepoints_per_epoch = metadata['timepoints_per_epoch']
    sampling_rate = metadata['sampling_rate']
    channel_names = metadata['channel_names']
    
    # Split into epochs based on usable epochs (not original total epochs)
    reconstructed_epochs_data = []
    for i in range(usable_epochs):
        start_idx = i * timepoints_per_epoch
        end_idx = (i + 1) * timepoints_per_epoch
        if end_idx <= reconstructed_data.shape[0]:
            epoch_data = reconstructed_data[start_idx:end_idx].T  # (channels, timepoints)
            reconstructed_epochs_data.append(epoch_data)
    
    reconstructed_epochs_data = np.array(reconstructed_epochs_data)
    
    # Create new epochs object with appropriate channel names
    if metadata['keep_256_channels'] == False:
        # For 256-channel mode, use original channel names (but model still outputs 19 channels)
        # Use the 19 standard names since that's what the model outputs
        standard_19_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", 
                           "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
        info = mne.create_info(ch_names=standard_19_names, sfreq=sampling_rate, ch_types='eeg')
    else:
        # For 19-channel mode, use the standard names
        info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types='eeg')
    
    reconstructed_epochs = mne.EpochsArray(reconstructed_epochs_data, info)
    
    # Copy events from original, but only for usable epochs
    original_events = original_epochs.events[:usable_epochs]
    reconstructed_epochs.events = original_events
    reconstructed_epochs.event_id = original_epochs.event_id
    
    return reconstructed_epochs


def process_single_file(file_path, output_dir, model, device, keep_256_channels):
    """
    Process a single FIF file through CBraMod.
    
    Args:
        file_path (str): Path to input FIF file
        output_dir (str): Directory to save output files
        model (CBraMod): Loaded CBraMod model
        device (torch.device): Device for inference
        keep_256_channels (bool): Whether to preserve original 256 channels
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nProcessing: {file_path}")
        
        # Load original epochs
        original_epochs = mne.read_epochs(file_path)
        print(f"Original data shape: {original_epochs.get_data().shape}")
        
        # Preprocess and run inference
        preprocessed_data, metadata = preprocess_data(file_path, keep_256_channels)
        output = inference(model, preprocessed_data, device)
        
        print(f"Usable epochs: {metadata['usable_epochs']}/{metadata['total_epochs']}")
        
        # Convert back to FIF format
        reconstructed_epochs = reconstruct_to_fif(output, original_epochs, metadata)
        original_epochs_preprocessed = reconstruct_to_fif(preprocessed_data, original_epochs, metadata)
        
        # Save reconstructed data
        subject_id = Path(file_path).parent.parent.parent.name
        session_num = Path(file_path).parent.parent.name
        
        # Create output directory with electrode count
        n_electrodes = 256 if keep_256_channels else 19
        output_subject_dir = Path(output_dir) / f"{n_electrodes}ch" / subject_id / session_num
        output_subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Save reconstructed file
        recon_filename = f"{subject_id}_{session_num}_task-rs_acq-01_epo_recon.fif"
        recon_path = output_subject_dir / recon_filename
        reconstructed_epochs.save(str(recon_path), overwrite=True)
        
        # Save original file
        orig_filename = f"{subject_id}_{session_num}_task-rs_acq-01_epo_orig.fif"
        orig_path = output_subject_dir / orig_filename
        original_epochs_preprocessed.save(str(orig_path), overwrite=True)
        
        print(f"Saved reconstructed: {recon_path}")
        print(f"Saved original: {orig_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch inference with CBraMod")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be processed without running inference")
    parser.add_argument("--keep_256", default = False, action="store_true",
                       help="Keep 256 channels instead of using 19 combined electrodes")
    parser.add_argument("--subjects", nargs="+", default=["all"],
                       help="Subject IDs to process (default: all)")
    parser.add_argument("--input_dir", default="/data/project/eeg_foundation/data/control/fifdata_256/nice_epochs_rs",
                       help="Input directory containing FIF files")
    parser.add_argument("--output_dir", default="/data/project/eeg_foundation/data/CbraMod/fif_data_control_rs",
                       help="Output directory for reconstructed files")
    
    args = parser.parse_args()
    
    # Determine electrode mode
    keep_256_channels = args.keep_256
    if keep_256_channels:
        print("Mode: 256-channel input with 19-channel reconstruction (preserving original 256 channels)")
    else:
        print("Mode: 19-channel input and reconstruction")
    
    # Setup device and model (only if not dry run)
    if not args.dry_run:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        model_path = '/home/triniborrell/home/projects/CBraMod/pretrained_weights/pretrained_weights.pth'
        model = load_pretrained_model(model_path, device)
    
    # Find all files to process
    all_files = find_all_fif_files(args.input_dir)
    print(f"Found {len(all_files)} FIF files")
    
    # Filter by subject if specified
    if args.subjects != ["all"]:
        all_files = [(s, sess, path) for s, sess, path in all_files if s in args.subjects]
    
    print(f"Found {len(all_files)} files to process")
    print(f"Mode: {'256' if keep_256_channels else '256→19'} electrode processing")
    
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Files that would be processed:")
        print("="*60)
        for subject_id, session_num, file_path in all_files:
            n_electrodes = 256 if keep_256_channels else 19
            output_subject_dir = Path(args.output_dir) / f"{n_electrodes}ch" / subject_id / session_num
            recon_path = output_subject_dir / f"{subject_id}_{session_num}_task-rs_acq-01_epo_recon.fif"
            orig_path = output_subject_dir / f"{subject_id}_{session_num}_task-rs_acq-01_epo_orig.fif"
            print(f"\nSubject: {subject_id}, Session: {session_num}")
            print(f"  Input:  {file_path}")
            print(f"  Output: {recon_path}")
            print(f"  Orig:   {orig_path}")
            print(f"  Mode:   {n_electrodes} channels")
        print("\n" + "="*60)
        print("DRY RUN COMPLETE - No files were processed")
        print("="*60)
        return
    
    # Process files
    successful = 0
    failed = 0
    
    for subject_id, session_num, file_path in all_files:
        if process_single_file(file_path, args.output_dir, model, device, keep_256_channels):
            successful += 1
        else:
            failed += 1
    
    print(f"\n" + "="*50)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print("="*50)


if __name__ == "__main__":
    main()
