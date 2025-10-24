import numpy as np
import torch
import tqdm
import pandas as pd
from pathlib import Path

from M1 import run_admm

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testing_set_dir = Path("../../testing_set")
    output_file = "../results/M1/rho_sweep_results.csv"
    
    # Rho values to test
    rho_values = [1] + list(range(5, 101, 5))
    
    # Number of trials for stability
    num_trials = 5
    
    print(f"Using device: {device}")
    print(f"Number of trials per rho: {num_trials}")
    print(f"Rho values: {rho_values}")
    
    # Load testing images
    testing_files = sorted(list(testing_set_dir.glob("*")))
    print(f"Found {len(testing_files)} testing images")
    
    # Store results
    results = []
    
    # Loop through rho values
    for rho in rho_values:
        print(f"\n{'='*60}")
        print(f"Testing rho = {rho}")
        print(f"{'='*60}")
        
        # Run multiple trials
        for trial in range(num_trials):
            print(f"\nTrial {trial + 1}/{num_trials}")
            
            trial_snr_list = []
            trial_ssim_list = []
            
            # Process each test image
            for img_file in tqdm.tqdm(testing_files, desc=f"Processing images"):
                _, _, snr, ssim = run_admm(img_file, device, rho=rho, max_iter=10000)
                trial_snr_list.append(snr)
                trial_ssim_list.append(ssim)
            
            # Store trial results
            avg_snr = np.mean(trial_snr_list)
            avg_ssim = np.mean(trial_ssim_list)
            
            results.append({
                'rho': rho,
                'trial': trial + 1,
                'avg_snr': avg_snr,
                'avg_ssim': avg_ssim,
                'std_snr': np.std(trial_snr_list),
                'std_ssim': np.std(trial_ssim_list)
            })
            
            print(f"  Trial {trial + 1} - SNR: {avg_snr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total records: {len(df)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
