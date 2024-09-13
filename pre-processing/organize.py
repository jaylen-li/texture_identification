import os
import shutil

# Define base paths
base_path = r'D:\\project_geo\\code_test_combined\\Cropped32_32'
new_base_path = r'D:\\project_geo\\code_test_combined\\organized32_32_updated'

# Ensure the new base path exists
os.makedirs(new_base_path, exist_ok=True)

# Iterate through each tile folder (e.g., "Tile 1")
for tile_name in os.listdir(base_path):
    tile_path = os.path.join(base_path, tile_name)
    if not os.path.isdir(tile_path):
        continue
    
    # Iterate through each class folder (e.g., "Building")
    for label in os.listdir(tile_path):
        label_path = os.path.join(tile_path, label)
        if not os.path.isdir(label_path):
            continue
        
        # Ensure the new class folder exists in the new base path
        new_label_path = os.path.join(new_base_path, label)
        os.makedirs(new_label_path, exist_ok=True)
        
        # Iterate over each image in the class folder
        for patch_file in os.listdir(label_path):
            if patch_file.endswith('.png'):  
                old_patch_path = os.path.join(label_path, patch_file)
                
                # The filename is already structured as tile1_001_land_(unpaved_area)_x352_y192
                # Split using a custom method to handle special characters
                parts = patch_file.replace('.png', '').split('_')
                
                # Handle case where label may contain parentheses or spaces
                tile = parts[0] 
                image_number = parts[1]  
                
                # Handle labels that may have parentheses and spaces
                label_parts = []
                coords_start_index = -1  # To track where coordinates (x, y) start
                for i, part in enumerate(parts[2:]):
                    if part.startswith('x') and 'y' in parts[i + 3]:  # Coordinates part starts here
                        coords_start_index = i + 2
                        break
                    else:
                        label_parts.append(part)
                
                label_name = '_'.join(label_parts)  # Join label parts
                coords = '_'.join(parts[coords_start_index:])  # Join x and y coordinates
                
                # Create the new file name and path
                new_patch_filename = f"{tile}_{image_number}_{label_name}_{coords}.png"
                new_patch_path = os.path.join(new_label_path, new_patch_filename)
                
                # Move the file
                shutil.move(old_patch_path, new_patch_path)
                print(f"Moved: {old_patch_path} -> {new_patch_path}")
            else:
                print(f"Unexpected file type: {patch_file}")

print("File restructuring complete.")
