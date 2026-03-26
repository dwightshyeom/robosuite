import h5py
import imageio
import numpy as np
from robosuite.controllers import load_composite_controller_config
from scripts.teleoperation.cdp.cube_pnp_cdp_tele import CubePlaceCDP # Make sure to import a custom environment

def render_custom_video(hdf5_path, video_path):
    '''
    Renders a video from a custom dataset of demonstrations.
    This example assumes the dataset is in HDF5 format with a specific structure,
    but you can modify it to fit your dataset's format.
    Args:
        hdf5_path (str): Path to the HDF5 dataset containing demonstrations.
        video_path (str): Path where the output video will be saved.
    '''
    print("Booting up environment...")
    
    # 1. Initialize your specific environment WITH the offscreen renderer ON
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = CubePlaceCDP(
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,            # No on-screen window needed
        has_offscreen_renderer=True,   # YES to offscreen for video
        use_camera_obs=True,           # YES to camera obs
        camera_names="frontview",     
        camera_heights=512,           
        camera_widths=512,            
        control_freq=20,
        ignore_done=True,
    )

    # 2. Open your dataset
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    print(f"Found {len(demos)} demonstrations.")

    writer = imageio.get_writer(video_path, fps=20)

    for ep in demos:
        print(f"Rendering {ep}...")
        
        # 3. Get the starting state and all actions for this demonstration
        states = f[f"data/{ep}/states"][()]
        actions = f[f"data/{ep}/actions"][()]
        
        # 4. Reset env and force it to the exact starting state
        env.reset()
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()
        
        # 5. Play the actions back open-loop and capture the frames
        for action in actions:
            obs, _, _, _ = env.step(action)
            frame = np.flipud(obs["frontview_image"])
            writer.append_data(frame)

    writer.close()
    f.close()
    print(f"Done! Video successfully saved to {video_path}")

if __name__ == "__main__":
    # Point this to your generated dataset!
    dataset_file = "/home/arclab/workspace/robosuite/custom_dataset/cube_pnp_cdp/demo_cubepnp_goal_120.hdf5"
    output_video = "./my_teleop_video.mp4"
    
    render_custom_video(dataset_file, output_video)