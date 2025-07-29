import torch
from torch.utils.data import DataLoader
from image_dataset import ImageDataset

# Set dataset paths (UPDATE THESE)
distorted_data_dir = "/workspaces/spiking-models-for-TSC/datasets/train-distorted"
clean_data_dir = "/workspaces/spiking-models-for-TSC/datasets/train-clean"

# Create dataset instance
dataset = ImageDataset(distorted_data_dir, clean_data_dir)

# Check dataset length
print(f"Total image pairs in dataset: {len(dataset)}")

# Create DataLoader for batching
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get a single batch
try:
    distorted_batch, clean_batch = next(iter(dataloader))
    print(f"Distorted batch shape: {distorted_batch.shape}")  # Expect [batch_size, 3, 128, 128]
    print(f"Clean batch shape: {clean_batch.shape}")  # Expect [batch_size, 3, 128, 128]
    print("Image dataset loading works correctly!")
except Exception as e:
    print(f"Error loading dataset: {e}")
