import numpy as np
import os
import shutil

# Load the .npy file
file_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/datasets/image_paths/LYHM.npy'
data = np.load(file_path, allow_pickle=True).item()
dataset_path = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/datasets/arcface/LYHM/arcface_input'

# Define the destination directory
destination_dir = '/shared/storage/cs/staffstore/ps1510/Tutorial/Dataset/LYHM_all'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate through the dictionary and copy each image
for subject_name, (image_list, _) in data.items():
    for image_path in image_list:
        # Extract the image name
        image_name = os.path.basename(image_path)
        
        image_path = os.path.join(dataset_path, image_path)
        # Construct the new file name and destination path
        new_file_name = f"{subject_name}_{image_name}"
        destination_path = os.path.join(destination_dir, new_file_name)
        
        # Copy the image
        shutil.copy(image_path, destination_path)

# List the copied files to verify
copied_files = os.listdir(destination_dir)
copied_files