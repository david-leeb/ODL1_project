import numpy as np
import torch
import pandas as pd
from pathlib import Path
import time

from M1 import run_admm


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testing_set_dir = Path("../../testing_set")
    output_file = "../results/M1/time_results.csv"
    
    # Optimal parameters from sweep results
    rho = 20
    rel_tol = 5e-5
    rel_tol2 = 5e-5
    max_iter = 10000
    
    print(f"Using device: {device}")
    print(f"Optimal parameters:")
    print(f"  rho: {rho}")
    print(f"  rel_tol: {rel_tol}")
    print(f"  rel_tol2: {rel_tol2}")
    print(f"  max_iter: {max_iter}")
    
    # Load testing images
    testing_files = sorted(list(testing_set_dir.glob("*")))
    print(f"\nFound {len(testing_files)} testing images")
    
    # Warmup run to initialize CUDA and compile kernels
    print("\nRunning warmup...")
    _ = run_admm(
        testing_files[0], device,
        spread_spectrum=True,
        rho=rho,
        rel_tol=rel_tol,
        rel_tol2=rel_tol2,
        max_iter=max_iter
    )
    print("Warmup complete!")
    
    # Store results
    results = []
    
    print("\nProcessing images...\n")
    
    # Process each test image
    for i, img_file in enumerate(testing_files, 1):
        print(f"[{i}/{len(testing_files)}] Processing {img_file.name}...", end=" ")
        
        # Time the reconstruction
        start_time = time.time()
        
        _, _, snr, ssim = run_admm(
            img_file, device,
            spread_spectrum=False,
            rho=rho,
            rel_tol=rel_tol,
            rel_tol2=rel_tol2,
            max_iter=max_iter
        )
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results.append({
            'image': img_file.name,
            'snr': snr,
            'ssim': ssim,
            'time_seconds': elapsed_time
        })
        
        print(f"SNR: {snr:.2f} dB, SSIM: {ssim:.4f}, Time: {elapsed_time:.2f}s")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"\nSummary Statistics:")
    print(f"  Average SNR: {df['snr'].mean():.2f} ± {df['snr'].std():.2f} dB")
    print(f"  Average SSIM: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    print(f"  Average Time: {df['time_seconds'].mean():.2f} ± {df['time_seconds'].std():.2f} seconds")
    print(f"  Total Time: {df['time_seconds'].sum():.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
