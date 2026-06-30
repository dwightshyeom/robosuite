import h5py
import json
import os

# 1. Double-check this path! Make sure it points exactly to your recorded dataset.
dataset_path = "../robosuite/custom_dataset/demo_fruit_swap.hdf5" 

# Safety check
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"\n[!] Oops! Could not find {dataset_path}.\n[!] Please check your folder path and filename.")

# 2. Use 'r+' mode so it strictly edits the existing file instead of creating a blank one
with h5py.File(dataset_path, "r+") as f:
    
    # Extract the current embedded JSON metadata
    env_args_str = f["data"].attrs["env_args"]
    env_meta = json.loads(env_args_str)
    
    # Fix the robot list inside the nested "env_kwargs" dictionary!
    if "env_kwargs" in env_meta and "robots" in env_meta["env_kwargs"]:
        env_meta["env_kwargs"]["robots"] = ["Panda"]
        
    # (Fallback just in case it's also at the top level)
    if "robots" in env_meta:
        env_meta["robots"] = ["Panda"]
        
    # Overwrite the attribute with the corrected JSON string
    f["data"].attrs["env_args"] = json.dumps(env_meta)
    
    print("\n[+] New Metadata:")
    print(json.dumps(env_meta, indent=4))
    
print("\n[✓] HDF5 metadata successfully patched!")