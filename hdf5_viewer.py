import h5py

def check_hdf5_structure(file_path):
    """
    Prints the internal structure of an HDF5 file, including groups, datasets, 
    their shapes, datatypes, and any attached attributes.
    """
    def print_node_info(name, obj):
        # Calculate indentation based on depth in the file hierarchy
        indent = "    " * (name.count('/') + 1)
        node_name = name.split('/')[-1]

        # Check if the object is a Group (folder) or a Dataset (data array)
        if isinstance(obj, h5py.Group):
            print(f"{indent}📁 Group: {node_name}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}📊 Dataset: {node_name} | Shape: {obj.shape} | Type: {obj.dtype}")
        
        # Print metadata attributes for the current node, if any exist
        for attr_name, attr_value in obj.attrs.items():
            print(f"{indent}    🏷️ Attr: {attr_name} = {attr_value}")

    try:
        # Open the file in read-only mode ('r')
        with h5py.File(file_path, 'r') as f:
            print(f"🗄️ HDF5 File: {file_path}")
            
            # Print root-level attributes (metadata attached to the file itself)
            for attr_name, attr_value in f.attrs.items():
                print(f"    🏷️ Root Attr: {attr_name} = {attr_value}")
                
            # visititems recursively calls our function on every item in the file
            f.visititems(print_node_info)
            
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"❌ Error reading the file: {e}")

# --- How to use it ---
# Replace 'your_data.h5' with the actual path to your HDF5 file
if __name__ == "__main__":
    # file_to_check = "./custom_dataset/low_dim_boxpush_120.hdf5"
    file_to_check = "./custom_dataset/low_dim_cubepnp_goal.hdf5"
    # file_to_check = "./custom_dataset/demo.hdf5"
    check_hdf5_structure(file_to_check)

# def inspect_dataset(file_path, dataset_path):
#     """
#     Opens an HDF5 file and prints a preview of a specific dataset.
#     """
#     try:
#         with h5py.File(file_path, 'r') as f:
#             # 1. Check if the path exists in the file
#             if dataset_path not in f:
#                 print(f"❌ Error: The path '{dataset_path}' was not found in the file.")
#                 return

#             # 2. Access the dataset
#             dataset = f[dataset_path]
            
#             # 3. Load the data into a NumPy array using [:]
#             # This reads the entire dataset from the disk into memory
#             data_array = dataset[:]
            
#             print(f"✅ Successfully loaded: {dataset_path}")
#             print(f"📊 Shape: {data_array.shape}")
#             print(f"🗂️ Data Type: {data_array.dtype}\n")
            
#             # 4. Print a preview (the first 5 rows) so it doesn't flood your console
#             print("Preview of the first 5 rows:")
#             print(data_array[:10])
            
#     except Exception as e:
#         print(f"❌ Error reading the file or dataset: {e}")

# # --- How to use it ---
# if __name__ == "__main__":
#     file_to_check = "./custom_dataset/low_dim_boxpush_goal.hdf5"
    
#     # IMPORTANT: Adjust this path based on your exact file structure!
#     # If the printout you shared was inside a demo group, you need to include it.
#     # For example, if it's inside 'demo_9', the path is 'demo_9/actions'
#     # target_data = "demo_9/actions" # demo_goal.hdf5
#     target_data = "data/demo_2/goal_obs/push_distance"  # <-- Adjust this based on your file's structure

#     inspect_dataset(file_to_check, target_data)
