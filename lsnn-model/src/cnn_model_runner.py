import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from cnn import CNN
from patch_dataset import PatchDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preview_utils import save_patch_grid, reassemble_image_grid
import piq

#dual losses
criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()

# ------------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
train_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/train-distorted"
train_clean_path = "/workspaces/spiking-models-for-TSC/datasets/train-clean"
test_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/test-distorted"
test_clean_path = "/workspaces/spiking-models-for-TSC/datasets/test-clean"
output_dir = "/workspaces/spiking-models-for-TSC/datasets/cnn-output"
os.makedirs(output_dir, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# ------------------- Dataset -------------------
full_train_dataset = PatchDataset(train_distorted_path, train_clean_path, transform=transform)
val_ratio = 0.08
val_size = int(len(full_train_dataset) * val_ratio)
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_dataset = PatchDataset(test_distorted_path, test_clean_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ------------------- Model -------------------
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

# ------------------- Training -------------------
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

        #Clamp output to [0, 1] range
        output_clamped = torch.clamp(output, 0.0, 1.0)

        # Compute loss
        mse_loss = criterion_mse(output, clean)
        l1_loss = criterion_l1(output, clean)
        ssim_loss = 1 - piq.ssim(output_clamped, clean, data_range=1.0)

        # Combined
        loss = 0.6 * mse_loss + 0.3 * l1_loss + 0.1 * ssim_loss
        #loss = 0.2 * mse_loss + 0.4 * l1_loss + 0.4 * ssim_loss
        #loss = 0.8 * mse_loss + 0.1 * l1_loss + 0.1 * ssim_loss

        #ssim_loss = 1 - piq.ssim(output, clean, data_range=1.)  # Data is in [0, 1]
        #loss = 0.7 * criterion_mse(output, clean) + 0.2 * criterion_l1(output, clean) + 0.1 * ssim_loss
        
        #loss = 0.9 * criterion_mse(output, clean) + 0.1 * criterion_l1(output, clean)
        
        #loss = criterion(output, clean)

        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_loss.append(avg_train_loss)

    # ------------------- Validation -------------------
    model.eval()
    running_val_loss = 0.0
    sample_inputs, sample_outputs, sample_targets = [], [], []
    with torch.no_grad():
        for distorted, clean in val_loader:
            distorted, clean = distorted.to(device), clean.to(device)
            output = model(distorted) 
            print(f"[Epoch {epoch+1}] Output Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
            loss = 0.9 * criterion_mse(output, clean) + 0.1 * criterion_l1(output, clean)
            #loss = criterion(output, clean)
            running_val_loss += loss.item()
            sample_inputs.append(distorted.cpu())
            sample_outputs.append(output.cpu())
            sample_targets.append(clean.cpu())

    avg_val_loss = running_val_loss / len(val_loader)
    val_loss.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save reconstructed full image preview
    reassemble_image_grid(sample_inputs, sample_outputs, sample_targets, output_dir, step=epoch, patch_size=32)

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(output_dir, f"cnn_epoch_{epoch+1}.pth"))

print("Training complete.")


# #custom skeleton to prepare a LSNN model to eventually
# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from cnn import CNN

# from patch_dataset import PatchDataset
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils.preview_utils import save_patch_grid

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #paths
# train_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/train-distorted"
# train_clean_path = "/workspaces/spiking-models-for-TSC/datasets/train-clean"
# test_distorted_path = "/workspaces/spiking-models-for-TSC/datasets/test-distorted"
# test_clean_path = "/workspaces/spiking-models-for-TSC/datasets/test-clean"
# output_dir = "/workspaces/spiking-models-for-TSC/datasets/cnn-output"
# os.makedirs(output_dir, exist_ok=True)

# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# # data
# full_train_dataset = PatchDataset(train_distorted_path, train_clean_path, transform=transform)
# val_ratio = 0.08
# val_size = int(len(full_train_dataset) * val_ratio)
# train_size = len(full_train_dataset) - val_size


# #I'm not creating a separate validation directory here so I can split the training dataset into training and validation sets.
# train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# test_dataset = PatchDataset(test_distorted_path, test_clean_path, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# # le model
# model = CNN().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.0002)
# criterion = nn.MSELoss()

# #train
# n_epochs = 10
# train_loss = []
# val_loss = []

# for epoch in range(n_epochs):
#     model.train()
#     running_train_loss = 0.0

#     for distorted, clean in train_loader:
#         distorted, clean = distorted.to(device), clean.to(device)
#         optimizer.zero_grad()
#         output = model(distorted)
#         loss = criterion(output, clean)
#         loss.backward()
#         optimizer.step()
#         running_train_loss += loss.item()

#     avg_train_loss = running_train_loss / len(train_loader)
#     train_loss.append(avg_train_loss)

#     #Validation
#     model.eval()
#     running_val_loss = 0.0
#     with torch.no_grad():
#         for distorted, clean in val_loader:
#             distorted, clean = distorted.to(device), clean.to(device)
#             output = model(distorted)
#             loss = criterion(output, clean)
#             running_val_loss += loss.item()

#     avg_val_loss = running_val_loss / len(val_loader)
#     val_loss.append(avg_val_loss)

#     print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#     # Save sample output grid for inspection
#     save_patch_grid(distorted.cpu(), output.cpu(), clean.cpu(), output_dir, step=epoch, max_images=1)

#     #ave model checkpoint
#     torch.save(model.state_dict(), os.path.join(output_dir, f"cnn_epoch_{epoch+1}.pth"))

# print("Training complete :)")
