import os
import shutil
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

# Define the resize transformation
resize_transform = transforms.Resize((224, 224))

def process_and_save_npy(image_path, output_npy_path, output_image_path):
    # Load the image using Pillow
    pil_image = Image.open(image_path)

    # Resize the image to 224x224
    resized_image = resize_transform(pil_image)

    # Save the resized image to the output directory
    resized_image.save(output_image_path)

    # Convert the resized Pillow image to a NumPy array
    image_array = np.array(resized_image)

    # Create the ArcFace input blob using OpenCV's DNN module
    input_mean = 127.5
    input_std = 127.5
    blob = cv2.dnn.blobFromImage(image_array, 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)

    # Save the ArcFace input blob as a .npy file
    np.save(output_npy_path, blob)

def process_images_in_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # Add other image extensions if needed
                # Full path to the input image
                image_path = os.path.join(root, file)

                # Determine the relative path from the input folder
                relative_path = os.path.relpath(root, input_folder)

                # Full path to the output subfolder, mirroring the input folder structure
                output_subfolder = os.path.join(output_folder, relative_path)

                # Paths for output files
                output_npy_path = os.path.join(output_subfolder, os.path.splitext(file)[0] + '.npy')
                output_image_path = os.path.join(output_subfolder, file)

                # Ensure the output subfolder exists
                os.makedirs(output_subfolder, exist_ok=True)

                # Process the image, save the resized image, and save the .npy file
                process_and_save_npy(image_path, output_npy_path, output_image_path)

# Example usage
input_folder = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/contents/LYHM_8_16'
output_folder = '/shared/storage/cs/staffstore/ps1510/Tutorial/3d-super-resolution-Face-reconstruction/datasets/temp_arcface/LYHM_8_16'
output_folder = output_folder + '/arcface_input'
process_images_in_folder(input_folder, output_folder)
