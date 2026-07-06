import h5py
import numpy as np

# RAW dataset (Update this to your actual raw cube place dataset!)
raw_dataset_path = "../robosuite/custom_dataset/cube_pnp_cdp/demo_cubepnp_goal_120.hdf5"

# FINAL dataset (Update this to your dataset that has the observations extracted)
final_dataset_path = "../robosuite/custom_dataset/image_pnp_cdp_256.hdf5"

f_raw = h5py.File(raw_dataset_path, "r")
f_final = h5py.File(final_dataset_path, "a")

for ep in f_final["data"]:
    ep_grp_final = f_final["data"][ep]
    ep_grp_raw = f_raw["data"][ep]
    
    # 1. Read the smuggled attribute from the RAW file (UPDATED)
    if "condition" in ep_grp_raw.attrs:
        target_dist = ep_grp_raw.attrs["condition"]
    else:
        print(f"Warning: No condition found for {ep} in raw file.")
        continue
        
    # 2. Match the exact timesteps of this episode
    timesteps = ep_grp_final["actions"].shape[0]
    
    # 3. Create the flat condition array
    cond_array = np.full((timesteps, 1), target_dist, dtype=np.float32)

    # 4. Inject the condition array into the FINAL dataset under obs/condition (UPDATED)
    if "condition" in ep_grp_final["obs"]:
        del ep_grp_final["obs"]["condition"]
        
    ep_grp_final["obs"].create_dataset("condition", data=cond_array)
    
    print(f"[{ep}] Injected obs/condition -> Value: {target_dist} | Shape: {cond_array.shape}")
    
f_raw.close()
f_final.close()
print("\n[✓] Goal conditions successfully injected!")