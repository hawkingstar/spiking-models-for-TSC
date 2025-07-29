import torch
from utils.base_utils import log
from pyt_train_eval_lsnn_and_lsnn_nhdn_model import get_batches_of_x_y_from_patches

#run with python test_data_loader.py

# Set dataset path (Update with correct path)
patch_dir = "/path/to/your/image_patches"  # Change this to your actual dataset path

# Test data loading
print("Testing image patch data loading...")

# Try loading a small batch
batch_size = 4  # Small batch for quick verification
data_loader = get_batches_of_x_y_from_patches(is_train=True, patch_dir=patch_dir, batch_size=batch_size)

# Get one batch and check its shape
try:
    distorted_batch, clean_batch = next(data_loader)
    print(f"Distorted batch shape: {distorted_batch.shape}")  # Should be [batch_size, 1, 64, 64]
    print(f"Clean batch shape: {clean_batch.shape}")  # Should match distorted batch
    print("Data loading works correctly!")
except Exception as e:
    print(f"Error loading data: {e}")
