import os
from PIL import Image

def check_and_delete_unsupported_images(folder_path):
    """
    Check for unsupported image formats in the given folder and delete them.
    Args:
        folder_path (str): Path to the folder containing images.
    """
    # Supported image formats
    # supported_formats = ['jpeg', 'png', 'jpg', 'bmp', 'tiff']
    supported_formats = ['jpeg']
    deleted_files = []

    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                # Open the image file
                with Image.open(os.path.join(subdir, file)) as img:
                    # Check if the image format is not in supported formats
                    if img.format.lower() not in supported_formats:
                        raise IOError("Unsupported format")
            except Exception as e:
                # If an error occurs (unsupported format or corrupt file), delete the file
                os.remove(os.path.join(subdir, file))
                deleted_files.append(os.path.join(subdir, file))
                print(f"Deleted '{os.path.join(subdir, file)}' due to error: {e}")

    return deleted_files

# Replace 'your_dataset_path' with the actual path of your dataset
# For example: '/path/to/dataset/'
dataset_path = '/data/hujingbin/ResNet50/mindspore_resnet50_husky_labrador/dataset/train/Cat'
deleted_files = check_and_delete_unsupported_images(dataset_path)

print(f"Deleted {len(deleted_files)} unsupported or corrupt files.")

