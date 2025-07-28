#just eben's version of the patch dataset code
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

# Custom dataset class to handle the distorted and clean image patches
class PatchDataset(Dataset):
    def __init__(self, distorted_data_dir, clean_data_dir, transform=None):
        self.distorted_data_dir = distorted_data_dir
        self.clean_data_dir = clean_data_dir
       
        # Get list of distorted and clean files, sorted naturally
        # natsorted is how the grayscale noised files are able to match to the base truths
        self.data_files_distorted = natsorted([f for f in os.listdir(distorted_data_dir) if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png'))])
        self.data_files_clean = natsorted([f for f in os.listdir(clean_data_dir) if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png'))])
       
        self.transform = transform
       
        # Ensure there are equal numbers of distorted and clean patches
        assert len(self.data_files_distorted) == len(self.data_files_clean), "Number of distorted and clean patches must be the same."

    def __len__(self):
        return len(self.data_files_distorted)

    def __getitem__(self, idx):
        # Load distorted patch
        distorted_file_path = os.path.join(self.distorted_data_dir, self.data_files_distorted[idx])
        # print(distorted_file_path)
        distorted_image = Image.open(distorted_file_path).convert('L')  # Convert to grayscale
       
        # Load clean patch
        clean_file_path = os.path.join(self.clean_data_dir, self.data_files_clean[idx])
        clean_image = Image.open(clean_file_path).convert('L')  # Convert to grayscale
       
        # Apply transformations if provided
        if self.transform:
            distorted_image = self.transform(distorted_image)
            clean_image = self.transform(clean_image)
       
        return distorted_image, clean_image

#leaving out for hardcoding
# # Define the transform for preprocessing (resize, convert to tensor, etc.)
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Resize to match patch size
#     transforms.ToTensor()  # Convert image to PyTorch tensor
# ])

# # Load Train dataset
# distorted_data_dir = r'/home/workstation/Desktop/cloud_data/train_data_cloud_(EbNo2)/EbNo(2)'  # Using augmented version
# clean_data_dir = r'/home/workstation/Desktop/cloud_data/train_data_cloud_clean/clean'  # Updated for grayscale version
# patch_dataset = PatchDataset(distorted_data_dir, clean_data_dir, transform=transform)
# patch_loader = DataLoader(patch_dataset, batch_size=16, shuffle=False)

# print("Starting DataLoader loop...")

# # Example of iterating through the DataLoader
# for i, (distorted_image, clean_image) in enumerate(patch_loader):
#     print(f"Iteration {i}: Distorted Patch Shape: {distorted_image.shape}, Clean Patch Shape: {clean_image.shape}")
#     break
# print(f'No of Images:{len(patch_loader)}')



# import os
# import numpy as np
# from PIL import Image
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from natsort import natsorted

# # Custom dataset class to handle the distorted and clean image patches
# class PatchDataset(Dataset):
#     def __init__(self, distorted_data_dir, clean_data_dir, transform=None):
#         self.distorted_data_dir = distorted_data_dir
#         self.clean_data_dir = clean_data_dir
        
#         self.data_files_distorted = natsorted([f for f in os.listdir(distorted_data_dir) if f.endswith(('.jpg', '.JPG','.jpeg','.JPEG','.png'))])
#         # print("Distorted files:", self.data_files_distorted)
#         self.data_files_clean = natsorted([f for f in os.listdir(clean_data_dir) if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG', '.png'))])
#         # print("Clean files:", self.data_files_clean)
#         self.transform = transform
       
#         assert len(self.data_files_distorted) == len(self.data_files_clean), "Number of distorted and clean patches must be the same."

#     def __len__(self):
#         return len(self.data_files_distorted)

#     def __getitem__(self, idx):
#         # Load distorted patch
#         distorted_file_path = os.path.join(self.distorted_data_dir, self.data_files_distorted[idx])
#         distorted_image = Image.open(distorted_file_path).convert('RGB')
       
#         # Load clean patch
#         clean_file_path = os.path.join(self.clean_data_dir, self.data_files_clean[idx])
#         clean_image = Image.open(clean_file_path).convert('RGB')
       
#         # Apply transformations if provided
#         if self.transform:
#             distorted_image = self.transform(distorted_image)
#             clean_image = self.transform(clean_image)
#             # print(f'clean_image:{clean_image}')
       
#         return distorted_image, clean_image

# # Define the transform for preprocessing (resize, convert to tensor, normalize, etc.)
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Patch size, assuming each patch is 32x32
#     transforms.ToTensor()  # Convert image to PyTorch tensor
# ])

#following was from the ipynb, for testinga and work
# Load dataset
# distorted_data_dir = r'/home/workstation/Desktop/Image_IQ/CRN_with_Distorted_Image/Distorted_Patches_Train_EbNo_23_BO_30' # Using Augmented Version
# clean_data_dir = r'/home/workstation/Desktop/Image_IQ/CRN_with_Distorted_Image/CleanPatches_Train_RGB_Aug' # Using Augmented version
# patch_dataset = PatchDataset(distorted_data_dir, clean_data_dir, transform=transform)
# patch_loader = DataLoader(patch_dataset, batch_size=16, shuffle=False)

# print("Starting DataLoader loop...")

# for i, (distorted_image, clean_image) in enumerate(patch_loader):
#     print(f"Iteration {i}: Distorted Patch Shape: {distorted_image.shape}, Clean Patch Shape: {clean_image.shape}")
#     break
    
# Example of iterating through the DataLoader
# for distorted_image, clean_image in patch_loader:
    # print(f"length of distorted image:{len(distorted_image)}")
    # print(f'Distorted Patch Shape: {distorted_image.shape}, Clean Patch Shape: {clean_image.shape}')
