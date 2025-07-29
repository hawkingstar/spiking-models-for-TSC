

import zipfile
import os

zip_path = '/workspaces/spiking-models-for-TSC/added-datasets/Grayscale_Train_AWGNPatches(EbNo10).zip'  # Replace with your zip path
extract_to = 's3://codespace-machine-learning/datasets/train-grayscale/'  # Destination directory

os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        # kip directory entries
        if member.endswith('/'):
            continue
        # Extract file directly into the target folder, ignoring nested structure
        filename = os.path.basename(member)
        if not filename:
            continue
        source = zip_ref.open(member)
        target_path = os.path.join(extract_to, filename)
        with open(target_path, "wb") as target:
            with source as src:
                target.write(src.read())


print("All files uploaded.")
