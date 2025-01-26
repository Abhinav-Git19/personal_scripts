import os
import shutil
import json

def transfer_images(json_file, destination_folder):
    """
    Transfers images listed as values in the JSON file to the specified destination folder.

    :param json_file: Path to the JSON file containing the image paths
    :param destination_folder: Path to the folder where images should be copied
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the list of image_path2 values and transfer them
    for image_path2 in data.values():
        if os.path.exists(image_path2):
            # Copy the file to the destination folder
            shutil.copy(image_path2, destination_folder)
            print(f"Copied: {image_path2}")
        else:
            print(f"File not found: {image_path2}")

# Example usage
if __name__ == "__main__":
    json_input_file = "image_matches.json"  # Path to your JSON file
    destination_folder = "collect_photos"  # Path to the destination folder
    transfer_images(json_input_file, destination_folder)
