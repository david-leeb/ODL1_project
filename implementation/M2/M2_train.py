"""
Simple training script for M2 U-Net
"""

import torch
from M2 import M2Trainer, parse_args


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("M2 U-Net Training")
    print("="*60)
    print(f"Ground truth path: {args.ground_truth_path}")
    print(f"Dataset fraction: {args.dataset_fraction}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr_init}")
    print(f"Network depth: {args.depth}, channels: {args.n_ch}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Save interval: every {args.model_save_interval} epochs")
    print(f"Device: {device}")
    print("="*60)
    
    # Initialize trainer with args from parse_args
    trainer = M2Trainer(
        path_gdth=args.ground_truth_path,
        path_dirty=args.ground_truth_path.replace("training_set", "training_set_dirty"),
        dataset_fraction=args.dataset_fraction,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        n_ch_in=args.n_ch_in,
        n_ch_out=args.n_ch_out,
        n_ch=args.n_ch,
        depth=args.depth,
        lr_init=args.lr_init,
        model_save_path=args.model_save_path,
        model_save_interval=args.model_save_interval,
        verbose=True,
        verbose_interval=args.verbose_interval,
        device=device
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining complete!")
    print(f"Models saved in: {args.model_save_path}/")

if __name__ == "__main__":
    main()