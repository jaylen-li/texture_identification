import cv2
import numpy as np
import os
import json

def hex_to_rgb(hex_color):
    # Convert hex color to an RGB tuple
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def load_color_map(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        # Convert colors from hex to RGB and then reverse to BGR for OpenCV
        return {tuple(reversed(hex_to_rgb(cls['color']))): cls['title'] for cls in data['classes']}

def save_patch(patch, label, output_dir, tile, image_number, x, y):
    # Save the image patch to the corresponding label folder within the tile
    folder = os.path.join(output_dir, tile, label.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    filename = f'{tile.lower().replace(" ", "")}_{image_number}_{label.lower()}_x{x}_y{y}.png'
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, patch)

def classify_patch(patch, color_map, tolerance=None):
    # Classify the patch by checking if it meets the tolerance level for less data
    if len(patch.shape) < 3:
        return "Mixed_Colors"

    reshaped_patch = patch.reshape(-1, patch.shape[2])
    unique_colors, counts = np.unique(reshaped_patch, axis=0, return_counts=True)
    
    total_pixels = patch.shape[0] * patch.shape[1]
    
    for color, count in zip(unique_colors, counts):
        color_tuple = tuple(color)
        
        if color_tuple in color_map:
            class_label = color_map[color_tuple]
            
            if class_label == "Road" or "Vegetation":
                if tolerance is not None and count / total_pixels >= tolerance:
                    return "Road"
                else:
                    return "Mixed_Colors"
            elif class_label == "Building":
               
                return class_label
            else:
                # For other classes, check if all pixels are the same
                if len(unique_colors) == 1:
                    return class_label
                else:
                    return "Mixed_Colors"
    
    return "Mixed_Colors"


base_path = r'D:\project_geo\code_test'
output_base_dir = r'D:\project_geo\code_test_combined\Cropped32_32'
json_path = r'D:\project_geo\code_test\classesUpdated.json'
window_size = 32  
default_stride_size = 32  
road_stride_size = int(default_stride_size * 0.2)  # 20% of the default stride size for the "Road" class
building_stride_size = default_stride_size  

color_map = load_color_map(json_path)

# Iterate through each tile
for tile_number in range(1, 9):
    tile_name = f'Tile {tile_number}'
    images_dir = os.path.join(base_path, tile_name, 'images')
    masks_dir = os.path.join(base_path, tile_name, 'masks')
    
    # Process each image and its corresponding mask
    for mask_filename in os.listdir(masks_dir):
        if mask_filename.endswith('.png'):
            image_filename = mask_filename.replace('.png', '.jpg')
            image_number = mask_filename.split('_')[-1].split('.')[0]  # Extract image number
            mask_path = os.path.join(masks_dir, mask_filename)
            image_path = os.path.join(images_dir, image_filename)
            
            mask = cv2.imread(mask_path)
            image = cv2.imread(image_path)
            
            if mask is None or image is None:
                continue
            
            # Ensure the image and mask dimensions match
            if mask.shape[:2] != image.shape[:2]:
                continue

            # Process mask in windows
            for y in range(0, mask.shape[0] - window_size + 1, default_stride_size):
                for x in range(0, mask.shape[1] - window_size + 1, default_stride_size):
                    mask_patch = mask[y:y+window_size, x:x+window_size]
                    image_patch = image[y:y+window_size, x:x+window_size]
                    
                    if mask_patch.size == 0 or image_patch.size == 0:
                        continue
                    
                    # Classify the patch
                    label = classify_patch(mask_patch, color_map, tolerance=0.6 if "Road" in color_map.values() else None)

                    if label == "Road" or "Vegetation":
                        stride_size = road_stride_size
                    elif label == "Building":
                        stride_size = building_stride_size
                    else:
                        stride_size = default_stride_size

                    if label not in ["Mixed_Colors", "Unlabeled"]:
                        save_patch(image_patch, label, output_base_dir, tile_name, image_number, x, y)

print("Processing complete.")
