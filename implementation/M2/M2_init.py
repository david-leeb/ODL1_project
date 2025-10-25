from M2 import gen_backprojected_image

def main():
    """Generate backprojected images for training and testing"""

    train_gdth = "../../training_set"
    train_dirty = "../../training_set_dirty"
    test_gdth = "../../testing_set"
    test_dirty = "../../testing_set_dirty"
    isnr = 30.0
    
    print("="*60)
    print("M2 Dataset Initialization")
    print("="*60)
    print(f"ISNR: {isnr} dB")
    print("="*60)
    
    # Generate training set dirty images
    print("\n[1/2] Generating backprojected TRAINING images...")
    print(f"  Input:  {train_gdth}")
    print(f"  Output: {train_dirty}")
    gen_backprojected_image(
        path_gdth=train_gdth,
        path_dirty=train_dirty,
        isnr=isnr
    )
    print("  ✓ Training set complete!")
    
    # Generate test set dirty images
    print("\n[2/2] Generating backprojected TESTING images...")
    print(f"  Input:  {test_gdth}")
    print(f"  Output: {test_dirty}")
    gen_backprojected_image(
        path_gdth=test_gdth,
        path_dirty=test_dirty,
        isnr=isnr
    )
    print("  ✓ Testing set complete!")
    
    print("\n" + "="*60)
    print("Dataset initialization complete!")
    print("="*60)

if __name__ == "__main__":
    main()
