import zipfile
import boto3
import os

# --- Your Config ---
zip_path = '/workspaces/spiking-models-for-TSC/added-datasets/Grayscale_Train_AWGNPatches(EbNo10).zip'
bucket_name = 'codespace-machine-learning'
s3_prefix = 'datasets/train-grayscale/'  # S3 path prefix inside the bucket

# --- AWS Client Setup ---
s3 = boto3.client('s3')

# --- Extract and Upload ---
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        if member.endswith('/'):
            continue  # Skip folders
        filename = os.path.basename(member)
        if not filename:
            continue

        print(f"Uploading {filename}...")

        # Read file directly from zip
        with zip_ref.open(member) as source_file:
            s3_key = s3_prefix + filename  # Full S3 key
            s3.upload_fileobj(source_file, bucket_name, s3_key)

print("All files extracted and uploaded directly to S3.")
