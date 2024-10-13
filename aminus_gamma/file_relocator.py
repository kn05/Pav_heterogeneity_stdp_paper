import os
import shutil

from config import *

# Define source and destination directories
destination_dir = os.path.join(result_dir, "in_out_scatter_plot")

# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate over folders from data00 to data64
for i in range(64):
    folder_name = f"data{i:02d}"  # Format folder name as data00, data01, ..., data64
    folder_path = os.path.join(result_dir, folder_name)

    # Define the source file path
    file_name = "target_sdf.jpg"
    source_file_path = os.path.join(folder_path, file_name)

    # Define the new file name and destination path
    new_file_name = f"{folder_name}_in_out_scatter_plot.png"
    destination_file_path = os.path.join(destination_dir, new_file_name)

    # Move and rename the file
    if os.path.isfile(source_file_path):
        shutil.move(source_file_path, destination_file_path)
        print(f"Moved: {source_file_path} to {destination_file_path}")
    else:
        print(f"File not found: {source_file_path}")
