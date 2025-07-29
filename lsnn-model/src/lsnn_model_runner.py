#basically a copy of cnn_model_runner with different 
#model to run on
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from cnn import CNN

from lsnn_image_denoiser import LSNNImageDenoiser
from lsnn_image_denoiser import SpikeEncoder2D

from patch_dataset import PatchDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preview_utils import save_patch_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#paths
train_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/train-distorted"
train_clean_path = "/workspaces/spiking-models-for-TSC/datasets/train-clean"
test_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/test-distorted"
test_clean_path = "/workspaces/spiking-models-for-TSC/datasets/test-clean"
output_dir = "/workspaces/spiking-models-for-TSC/datasets/cnn-output"
os.makedirs(output_dir, exist_ok=True)

# Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# data
full_train_dataset = PatchDataset(train_distorted_path, train_clean_path, transform=transform)
val_ratio = 0.08
val_size = int(len(full_train_dataset) * val_ratio)
train_size = len(full_train_dataset) - val_size


#I'm not creating a separate validation directory here so I can split the training dataset into training and validation sets.
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = PatchDataset(test_distorted_path, test_clean_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# le model, lsnn based this time
model = LSNNImageDenoiser().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

#train
n_epochs = 10
train_loss = []
val_loss = []

for epoch in range(n_epochs):
    model.train()
    running_train_loss = 0.0

    for distorted, clean in train_loader:
        distorted, clean = distorted.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(distorted)

        #debugging output shape for the very first batch
        if epoch == 0 and batch_idx == 0:
            print("Input shape:", distorted.shape)
            print("Output shape:", output.shape)
            print("Target shape:", clean.shape)

        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    #Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for distorted, clean in val_loader:
            distorted, clean = distorted.to(device), clean.to(device)
            output = model(distorted)
            loss = criterion(output, clean)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_loss.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save sample output grid for inspection
    save_patch_grid(distorted.cpu(), output.cpu(), clean.cpu(), output_dir, step=epoch)

    #ave model checkpoint
    #changed the name to lsnn epochs
    torch.save(model.state_dict(), os.path.join(output_dir, f"lsnn_epoch_{epoch+1}.pth"))


print("Training complete :)")
