import os
import glob
import shutil
import json
from robosuite.src.utils.dataset_utils import gather_demonstrations_as_hdf5
from robosuite.controllers import load_composite_controller_config

# 1. Hardcoded path to your exact teleoperation session
tmp_directory = "/dev/shm/teleop_1774575087_9926684"

if not os.path.exists(tmp_directory):
    print(f"[!] Critical Error: The folder {tmp_directory} no longer exists.")
    exit()

print(f"[+] Scanning demonstrations in: {tmp_directory}")

# 2. PURGE THE BAD DATA
ep_folders = glob.glob(os.path.join(tmp_directory, "ep_*"))
valid_count = 0
deleted_count = 0

for ep_dir in ep_folders:
    condition_file = os.path.join(ep_dir, "model.xml")
    
    if os.path.exists(condition_file):
        valid_count += 1
    else:
        # If the user pressed 'n', 'q', or 'ESC', this file won't exist. Nuke the folder!
        shutil.rmtree(ep_dir)
        deleted_count += 1

print(f"  -> Found {valid_count} valid demonstrations.")
print(f"  -> Purged {deleted_count} discarded/incomplete demonstrations.")

if valid_count == 0:
    print("[!] No valid demonstrations found to compile. Exiting.")
    exit()

# 3. Rebuild the environment info JSON
controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
config = {
    "env_name": "PutKExactBlocks",
    "robots": ["Panda"],
    "controller_configs": controller_config,
}
env_info_json = json.dumps(config)

# 4. Compile ONLY the valid runs!
final_out_dir = "./custom_dataset"
print("\n[+] Starting clean compilation...")

try:
    gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
    print(f"\n[✓] MASSIVE SUCCESS! Your {valid_count} clean demonstrations have been rescued and compiled.")
except Exception as e:
    print(f"\n[!] Compilation failed: {e}")