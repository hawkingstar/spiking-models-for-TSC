import os
import torch
from torchvision.utils import save_image

def save_patch_grid(input_tensor, output_tensor, target_tensor, output_dir, step=0):
    os.makedirs(output_dir, exist_ok=True)
    input_imgs = input_tensor.view(-1, 1, 32, 32)
    output_imgs = output_tensor.view(-1, 1, 32, 32)
    target_imgs = target_tensor.view(-1, 1, 32, 32)

    grid = torch.cat([input_imgs, output_imgs, target_imgs], dim=3)  # B x 1 x 32 x 96

    for i, img in enumerate(grid):
        save_image(img, os.path.join(output_dir, f"sample_{step}_{i}.png"))
