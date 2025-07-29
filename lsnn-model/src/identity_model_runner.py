import os
import torch
import sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from image_dataset import ImageDataset  # Your earlier version
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preview_utils import save_patch_grid  # From earlier step

#SETTINGS 
DISTORTED_DIR = "/workspaces/spiking-models-for-TSC/datasets/train-distorted"
CLEAN_DIR = "/workspaces/spiking-models-for-TSC/datasets/train-clean"
OUTPUT_DIR = "/workspaces/spiking-models-for-TSC/datasets/identity-output"
BATCH_SIZE = 16

#TRANSFORm 
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

#DATASET & LOADER 
dataset = ImageDataset(DISTORTED_DIR, CLEAN_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

#iDENTITY MODEL LOOP 
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Running identity model...")

for step, (distorted, clean) in enumerate(dataloader):
    distorted_flat = distorted[:, 0, :, :].view(distorted.size(0), -1)
    clean_flat = clean[:, 0, :, :].view(clean.size(0), -1)

    #output = input
    output = distorted_flat.clone()

    # Save
    save_patch_grid(distorted_flat, output, clean_flat, output_dir=OUTPUT_DIR, step=step)

    print(f"Saved batch {step}")
    if step >= 2:  #only first 3 batches
        break

print("Done :)")