import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from patch_dataset import PatchDataset  # Assuming you saved the class in patch_dataset.py

#run with python test_patch_dataset.py

#should output:
#Total patches in dataset: 4000  # (If you had 1000 images, 4 patches each)
# Distorted batch shape: torch.Size([4, 1, 64, 64])
# Clean batch shape: torch.Size([4, 1, 64, 64])


# Define paths (UPDATE THESE)
distorted_data_dir = "jaja"
clean_data_dir = "jaja"

# Define transformation (convert images to tensors)
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
])

# Initialize dataset
dataset = PatchDataset(distorted_data_dir, clean_data_dir, transform=transform)

# Check dataset length
print(f"Total patches in dataset: {len(dataset)}")  # Should be 4× the number of images

# Create DataLoader for batching
batch_size = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Get a single batch
distorted_batch, clean_batch = next(iter(dataloader))

# Print shapes (Should be [batch_size, 1, 64, 64])
print(f"Distorted batch shape: {distorted_batch.shape}")  
print(f"Clean batch shape: {clean_batch.shape}")  
