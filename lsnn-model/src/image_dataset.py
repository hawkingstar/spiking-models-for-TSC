import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    """
    Dataset class to load paired distorted and clean images.
    """
    def __init__(self, distorted_data_dir, clean_data_dir, transform=None):
        """
        Args:
            distorted_data_dir (str): Path to distorted images.
            clean_data_dir (str): Path to clean images.
            transform (callable, optional): Transformations to apply to images.
        """
        self.distorted_data_dir = distorted_data_dir
        self.clean_data_dir = clean_data_dir

        # Get sorted list of image files
        self.data_files_distorted = sorted([f for f in os.listdir(distorted_data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        self.data_files_clean = sorted([f for f in os.listdir(clean_data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

        assert len(self.data_files_distorted) == len(self.data_files_clean), "Number of distorted and clean images must be the same."

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),  # Resize all images to a fixed size
            transforms.ToTensor()  # Convert images to PyTorch tensors
        ])

    def __len__(self):
        return len(self.data_files_distorted)

    def __getitem__(self, idx):
        # Load distorted image
        distorted_path = os.path.join(self.distorted_data_dir, self.data_files_distorted[idx])
        distorted_image = Image.open(distorted_path).convert('RGB')

        # Load clean image
        clean_path = os.path.join(self.clean_data_dir, self.data_files_clean[idx])
        clean_image = Image.open(clean_path).convert('RGB')

        # Apply transformations
        distorted_image = self.transform(distorted_image)
        clean_image = self.transform(clean_image)

        return distorted_image, clean_image
