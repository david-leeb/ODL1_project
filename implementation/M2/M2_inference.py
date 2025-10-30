"""
Simple inference script for M2 U-Net
Run M2_init.py first to generate backprojected images
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
import skimage.metrics as metrics

from M2 import UNet


def rsnr(x_rec, x_true):
    """Calculate reconstruction SNR"""
    return 20 * np.log10(np.linalg.norm(x_true.flatten()) / 
                         np.linalg.norm(x_true.flatten() - x_rec.flatten()))


def run_inference(test_gdth, test_dirty, model_path, n_ch=64, depth=3, device=None, verbose=True):
    """
    Run inference on test set and compute metrics with timing
    
    Args:
        test_gdth: Path to ground truth test images
        test_dirty: Path to dirty (backprojected) test images
        model_path: Path to trained model checkpoint
        n_ch: Number of channels in first layer
        depth: Depth of U-Net
        device: Device to use (cuda or cpu)
        verbose: Print progress information
    
    Returns:
        pandas DataFrame with results for each image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
        print(f"Loading model from {model_path}...")
    
    # Load model
    model = UNet(n_ch_in=1, n_ch_out=1, n_ch=n_ch, depth=depth)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    if verbose:
        print("Model loaded!")
    
    # Get all test files
    gdth_files = sorted(glob.glob(os.path.join(test_gdth, "*.tiff")))
    
    # Warmup run
    if verbose:
        print("\nRunning warmup...")
    dummy_gdth = np.array(Image.open(gdth_files[0])).astype(np.float32)
    dummy_dirty_file = os.path.join(test_dirty, os.path.basename(gdth_files[0]))
    dummy_dirty = np.array(Image.open(dummy_dirty_file)).astype(np.float32)
    dummy_tensor = torch.tensor(dummy_dirty, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _ = model(dummy_tensor)
    if verbose:
        print("Warmup complete!")
        print("\nProcessing images...\n")
    
    # Store results
    results = []
    
    # Run inference and compute metrics
    with torch.no_grad():
        for i, gdth_file in enumerate(gdth_files, 1):
            if verbose:
                print(f"[{i}/{len(gdth_files)}] Processing {os.path.basename(gdth_file)}...", end=" ")
            
            # Load ground truth
            gdth = np.array(Image.open(gdth_file)).astype(np.float32)
            
            # Load dirty (backprojected) image
            dirty_file = os.path.join(test_dirty, os.path.basename(gdth_file))
            dirty = np.array(Image.open(dirty_file)).astype(np.float32)
            
            # Metrics for backprojected image
            snr_bp = rsnr(dirty, gdth)
            ssim_bp = metrics.structural_similarity(gdth, dirty, data_range=gdth.max() - gdth.min())
            
            # Run inference with timing
            dirty_tensor = torch.tensor(dirty, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            start_time = time.time()
            reconstructed_tensor = model(dirty_tensor)
            elapsed_time = time.time() - start_time
            
            reconstructed = reconstructed_tensor.squeeze().cpu().numpy()
            
            # Metrics for reconstructed image
            snr_rec = rsnr(reconstructed, gdth)
            ssim_rec = metrics.structural_similarity(gdth, reconstructed, data_range=gdth.max() - gdth.min())
            
            # Store results
            result = {
                'image': os.path.basename(gdth_file),
                'snr_backprojected': snr_bp,
                'ssim_backprojected': ssim_bp,
                'snr_reconstructed': snr_rec,
                'ssim_reconstructed': ssim_rec,
                'time_seconds': elapsed_time
            }
            results.append(result)
            
            result['original'] = gdth
            result['dirty'] = dirty
            result['reconstructed'] = reconstructed
            
            if verbose:
                print(f"SNR: {snr_rec:.2f} dB, SSIM: {ssim_rec:.4f}, Time: {elapsed_time:.4f}s")
    
    return pd.DataFrame(results)

def main():
    """Main inference function - measures pure inference time"""
    # Hardcoded parameters
    test_dirty = "../../testing_set_dirty"
    model_path = "../results/M2/models/default_bs16/model_epoch_50.ckpt"
    output_file = "../results/M2/time_results.csv"
    n_ch = 64
    depth = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("M2 U-Net Inference Timing")
    print("="*60)
    print(f"Test dirty images: {test_dirty}")
    print(f"Model checkpoint: {model_path}")
    print(f"Network depth: {depth}, channels: {n_ch}")
    print(f"Device: {device}")
    print("="*60)
    
    # Check that dirty images exist
    if not os.path.exists(test_dirty):
        print(f"\nError: Dirty test images not found at {test_dirty}")
        print("Please run M2_init.py first to generate backprojected images.")
        return
    
    # Load model
    print(f"\nLoading model...")
    model = UNet(n_ch_in=1, n_ch_out=1, n_ch=n_ch, depth=depth)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Get all test files
    dirty_files = sorted(glob.glob(os.path.join(test_dirty, "*.tiff")))
    print(f"Found {len(dirty_files)} test images")
    
    # Warmup run
    print("\nRunning warmup...")
    dummy_dirty = np.array(Image.open(dirty_files[0])).astype(np.float32)
    dummy_tensor = torch.tensor(dummy_dirty, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        _ = model(dummy_tensor)
    print("Warmup complete!")
    
    # Run pure inference with timing
    print("\nRunning inference loop...\n")
    
    results = []
    with torch.no_grad():
        for i, dirty_file in enumerate(dirty_files, 1):
            # Load dirty image
            dirty = np.array(Image.open(dirty_file)).astype(np.float32)
            dirty_tensor = torch.tensor(dirty, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            # Time individual inference
            start_time = time.time()
            _ = model(dirty_tensor)
            elapsed_time = time.time() - start_time
            
            results.append({
                'image': os.path.basename(dirty_file),
                'time_seconds': elapsed_time
            })
            
            if i % 5 == 0:
                print(f"Processed {i}/{len(dirty_files)} images")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Print timing results
    print("\n" + "="*60)
    print("TIMING RESULTS")
    print("="*60)
    print(f"Results saved to: {output_file}")
    print(f"\nTotal images: {len(df)}")
    print(f"Total time: {df['time_seconds'].sum():.4f} seconds")
    print(f"Average time per image: {df['time_seconds'].mean():.4f} ± {df['time_seconds'].std():.4f} seconds")
    print(f"Min time: {df['time_seconds'].min():.4f} seconds")
    print(f"Max time: {df['time_seconds'].max():.4f} seconds")
    print(f"Throughput: {len(df)/df['time_seconds'].sum():.2f} images/second")
    print("="*60)

if __name__ == "__main__":
    main()
