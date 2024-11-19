import os
from PIL import Image
import random

int_to_labels = {
    0: 'Fishing',
  1: 'Transport',
  2: 'Speedboat',
  3: 'Voilier',
  4: 'Military',
  5: 'Service',
}

int_to_labels_2 = {
    6: 'Mouvement',
    7: 'Stationnaire',}
def convert_to_dota(annotation, img_width, img_height):
    """
    Converts a normalized annotation to DOTA format for a specific image size.
    
    Args:
    - annotation (list): List of values in the format:
                         [label, difficulty, x1, y1, x2, y2, x3, y3, x4, y4]
    - img_width (int): Width of the image.
    - img_height (int): Height of the image.
    
    Returns:
    - dota_annotation (str): Annotation in the DOTA format as a string.
    """
    label = annotation[0]
    label_2 = annotation[1]
    
    # Convert normalized coordinates to absolute pixel values
    x1 = annotation[2] * img_width
    y1 = annotation[3] * img_height
    x2 = annotation[4] * img_width
    y2 = annotation[5] * img_height
    x3 = annotation[6] * img_width
    y3 = annotation[7] * img_height
    x4 = annotation[8] * img_width
    y4 = annotation[9] * img_height
    
    return f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} {int_to_labels[int(label)]} {int_to_labels_2[int(label_2)]}  {0}"


def process_annotations(img_dir, annot_dir, output_dir):
    """
    Processes all images and annotations, converts them to DOTA format, and saves them in the output directory.
    
    Args:
    - img_dir (str): Directory containing image files.
    - annot_dir (str): Directory containing corresponding annotation text files.
    - output_dir (str): Directory to save the converted DOTA format annotations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_file)
        
        # Ensure it's an image file we can read
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
            continue

        # Get image size
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        # Corresponding annotation file (assuming same name but with .txt extension)
        annot_file = os.path.splitext(img_file)[0] + '.txt'
        annot_path = os.path.join(annot_dir, annot_file)

        if not os.path.exists(annot_path):
            print(f"Warning: Annotation file {annot_file} not found for image {img_file}")
            continue

        # Read and convert annotations
        dota_annotations = []
        with open(annot_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 10:
                    print(f"Invalid format in {annot_file}, skipping line: {line}")
                    continue
                
                # Convert parts to appropriate types
                annotation = [int(parts[0]), int(parts[1])] + [float(p) for p in parts[2:]]
                
                # Convert to DOTA format
                dota_annotation = convert_to_dota(annotation, img_width, img_height)
                dota_annotations.append(dota_annotation)

        # Save DOTA annotations to output directory
        output_path = os.path.join(output_dir, annot_file)
        with open(output_path, 'w') as f_out:
            for dota_annotation in dota_annotations:
                f_out.write(dota_annotation + '\n')
        
        print(f"Processed {img_file} and saved DOTA annotations to {output_path}")

import os
import numpy as np
from PIL import Image

def calculate_mean_std(directory_path):
    means = []
    stds = []


    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Load image
        image = Image.open(file_path).convert("RGB")  # Ensure 3 channels
        image_array = np.array(image)  # Raw pixel values [0, 255]
        m = np.mean(image_array, axis=(0, 1))
        s = np.std(image_array, axis=(0, 1))

        means.append(m)
        stds.append(s)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std

directory_path = "data/dota_ml/Images"  # Replace with your directory
mean, std = calculate_mean_std(directory_path)
print(f"Mean: {mean}")
print(f"Std: {std}")




# Example usage
img_dir = 'data/dota_ml/Images'          # Replace with your image directory path
annot_dir = 'data/multilabel'    # Replace with your annotation directory path
output_dir = 'data/dota_ml/label'       # Replace with your output directory path

#process_annotations(img_dir, annot_dir, output_dir)