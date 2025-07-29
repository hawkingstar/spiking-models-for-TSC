import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import math

def reassemble_image_grid(inputs, outputs, targets, output_dir, step=0, patch_size=32):
    """
    Reassembles image patches into a grid and saves as a single image.
    
    Args:
        inputs: List of tensors of shape [B, 1, 32, 32] (noisy patches)
        outputs: List of tensors of same shape (denoised patches)
        targets: List of tensors of same shape (clean patches)
        output_dir: directory to save image
        step: current epoch number
        patch_size: assumed patch size
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Concatenate all batches along dim 0
    inputs = torch.cat(inputs, dim=0)
    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    # Limit to a square number of patches for visualization
    num_patches = min(64, inputs.size(0), outputs.size(0), targets.size(0))
    side = int(math.sqrt(num_patches))
    inputs = inputs[:side*side]
    outputs = outputs[:side*side]
    targets = targets[:side*side]
    # Stack each (input | output | target) horizontally per row
    strips = []
    for i in range(inputs.size(0)):
        strip = torch.cat([inputs[i], outputs[i], targets[i]], dim=2)
        strips.append(strip)

    # Stack vertically
    grid = torch.stack(strips, dim=0)  # [B, 1, 32, 96]
    grid = torchvision.utils.make_grid(grid, nrow=side, padding=2, pad_value=1)

    # Save
    save_image(grid, os.path.join(output_dir, f"epoch_{step:02d}_reassembled.png"))

def save_patch_grid(input_tensor, output_tensor, target_tensor, output_dir, step=0, patch_size=(32, 32), max_images=4):
    """
    Save a horizontal row of `max_images` triplets: (input | output | target).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clamp to the number of available samples
    max_images = min(max_images, input_tensor.size(0))

    rows = []
    for i in range(max_images):
        input_img = input_tensor[i].view(1, 1, *patch_size)
        output_img = output_tensor[i].view(1, 1, *patch_size)
        target_img = target_tensor[i].view(1, 1, *patch_size)
        triplet = torch.cat([input_img, output_img, target_img], dim=3)  # [1, 1, H, 3W]
        rows.append(triplet)

    grid = torch.cat(rows, dim=2)  # [1, 1, max_images*H, 3W]
    save_image(grid, os.path.join(output_dir, f"sample_{step}.png"))

    # input_imgs = input_tensor.view(-1, 1, 32, 32)
    # output_imgs = output_tensor.view(-1, 1, 32, 32)
    # target_imgs = target_tensor.view(-1, 1, 32, 32)

    # # upscale?
    # input_imgs = F.interpolate(input_imgs, scale_factor=upscale_factor, mode='nearest')
    # output_imgs = F.interpolate(output_imgs, scale_factor=upscale_factor, mode='nearest')
    # target_imgs = F.interpolate(target_imgs, scale_factor=upscale_factor, mode='nearest')

    # # concatenate horizontally
    # grid = torch.cat([input_imgs, output_imgs, target_imgs], dim=3)

    # for i, img in enumerate(grid):
    #     save_image(img, os.path.join(output_dir, f"sample_{step}_{i}.png"))


# def save_patch_grid(input_tensor, output_tensor, target_tensor, output_dir, step=0):
#     os.makedirs(output_dir, exist_ok=True)
#     input_imgs = input_tensor.view(-1, 1, 32, 32)
#     output_imgs = output_tensor.view(-1, 1, 32, 32)
#     target_imgs = target_tensor.view(-1, 1, 32, 32)

#     grid = torch.cat([input_imgs, output_imgs, target_imgs], dim=3)  # B x 1 x 32 x 96

#     for i, img in enumerate(grid):
#         save_image(img, os.path.join(output_dir, f"sample_{step}_{i}.png"))


