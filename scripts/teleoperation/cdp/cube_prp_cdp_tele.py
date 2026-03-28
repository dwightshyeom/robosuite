import os
import time
import json
import rclpy
import shutil
import numpy as np
from copy import deepcopy

# robosuite Imports
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.controllers.composite.composite_controller import WholeBody

# Custom Imports
from robosuite.src.utils.dataset_utils import gather_demonstrations_as_hdf5
from robosuite.src.utils.teleop_node_utils import TeleopNode
from robosuite.src.device.phantom import PhantomOmni

# Import the new environment!
from robosuite.environments.cdp.cube_prp_cdp import CubePickRotatePlaceCDP

# Main Data Collection Pipeline
if __name__ == "__main__":
    rclpy.init()
    node = TeleopNode()
    print("ROS 2 Started. Waiting for /arm/measured_cp...")

    # Wait for the first message from the Phantom Omni to ensure connection
    while node.master_position is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("[!] Phantom Omni connected.")

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

    # Environment Configuration for the HDF5 metadata
    config = {
        "env_name": "CubePlaceCDP",
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    env_info_json = json.dumps(config)

    # Initialize the custom Pick and Place environment
    env = CubePickRotatePlaceCDP(
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,            
        renderer='mjviewer',           
        has_offscreen_renderer=False,  
        use_camera_obs=False,          
        control_freq=20,
        ignore_done=True,              
    )

    # Wrap environment (DataCollection must wrap before Visualization)
    tmp_directory = f"/dev/shm/teleop_{str(time.time()).replace('.', '_')}"

    data_wrapper = DataCollectionWrapper(env, tmp_directory)
    env = VisualizationWrapper(data_wrapper)

    # Initialize the Phantom Omni device
    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)

    print("\n" + "="*50)
    print("CUBE PLACE DATA COLLECTION PIPELINE (CDP)")
    print("="*50)

    num_successful_demos = 0

    try:
        # OUTER LOOP: Manages Episodes
        while True:
            print(f"\n[Dataset: {num_successful_demos} Demos Saved]")
            print("Select target PLACE rotation for this demonstration:")
            print("  [1] 60 degrees")
            print("  [2] 120 degrees")
            print("  [3] 180 degrees")

            # Condition Input
            user_input = input("Enter choice (1/2/3) or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                break
                
            if user_input == '1':
                target_rotation = 60.0
            elif user_input == '2':
                target_rotation = 120.0
            elif user_input == '3':
                target_rotation = 180.0
            else:
                print("[!] Invalid input. Please enter 1, 2, or 3.")
                continue
            
            # Inject Condition into the environment before resetting
            env.set_target_angle(target_rotation)

            print(f"\n>>> Recording started! Pick the cube and place it exactly {target_rotation}m away. <<<")
            obs = env.reset()
            device.start_control()
            
            success_hold_count = 10 
            all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]

            # INNER LOOP: Teleoperation
            while True:
                start_time = time.time()
                rclpy.spin_once(node, timeout_sec=0.0)
                active_robot = env.robots[device.active_robot]
                
                input_ac_dict = device.input2action()
                
                # ESC manually aborts the loop
                if input_ac_dict is None:
                    print("\n[!] ESC pressed. Halting current demonstration.")
                    break 
                    
                action_dict = deepcopy(input_ac_dict)
                
                # NOTE: The forced gripper closure (`action_dict[f"{arm}_gripper"] = np.ones(...)`) 
                # has been removed here so you can freely open and close the gripper!
                
                # Arm movement parsing
                for arm in active_robot.arms:
                    if isinstance(active_robot.composite_controller, WholeBody): 
                        controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = active_robot.part_controllers[arm].input_type

                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]

                # Construct action vector properly
                env_action = []
                for i, robot in enumerate(env.robots):
                    if i == device.active_robot:
                        env_action.append(active_robot.create_action_vector(action_dict))
                    else:
                        env_action.append(robot.create_action_vector(all_prev_gripper_actions[i]))
                env_action = np.concatenate(env_action)
                
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

                obs, reward, done, info = env.step(env_action)
                env.render()
                
                # Success state machine
                if env._check_success():
                    print(f"  [~] Success condition met! Hold for {success_hold_count} more steps...")
                    success_hold_count -= 1
                else:
                    success_hold_count = 10 
                    
                # Auto-break when task is completed
                if success_hold_count <= 0:
                    print(f"\n[✓] Goal Reached! Cube successfully placed at {target_rotation}°.")
                    break 
                
                # Enforce 20Hz control
                diff = (1 / 20) - (time.time() - start_time)
                if diff > 0:
                    time.sleep(diff)

            # Sync array lengths (Robosuite naturally saves N actions and N+1 states)
            min_len = min(len(data_wrapper.states), len(data_wrapper.action_infos))
            data_wrapper.states = data_wrapper.states[:min_len]
            data_wrapper.action_infos = data_wrapper.action_infos[:min_len]

            # POST-DEMONSTRATION PROMPT
            save_input = input("Do you want to SAVE this demonstration? (y/n): ")
            
            if save_input.lower() == 'y':
                print("[+] Demonstration kept in temporary memory.")
                num_successful_demos += 1
                # Save the custom parameter for later HDF5 injection
                np.save(os.path.join(data_wrapper.ep_directory, "condition.npy"), np.array([target_rotation]))

            else:
                print("[-] Discarding data...")
                # Clear the data collection wrapper's internal memory
                env.states = []
                env.action_infos = []
                
                # Wipe the specific episode folder, but recreate the shell to prevent flush crashes!
                if hasattr(env, 'ep_directory') and os.path.exists(env.ep_directory):
                    shutil.rmtree(env.ep_directory)
                    os.makedirs(env.ep_directory, exist_ok=True) # <-- BROUGHT THIS BACK
                    
                print("    Data wiped. (Empty shell retained to prevent crash)")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        rclpy.shutdown()
        env.close() # <-- Robosuite does its final flush here

        # ==========================================================
        # GHOST HUNTER CLEANUP
        # Purge all empty/discarded folders before the compiler runs
        # ==========================================================
        import glob
        ep_folders = glob.glob(os.path.join(tmp_directory, "ep_*"))
        for ep_dir in ep_folders:
            condition_file = os.path.join(ep_dir, "condition.npy")
            if not os.path.exists(condition_file):
                shutil.rmtree(ep_dir) # Vaporize the ghost folder
        # ==========================================================

        print(f"\nCompiling {num_successful_demos} saved demonstrations into `.hdf5` dataset...")
        final_out_dir = "./custom_dataset" 
        
        if num_successful_demos > 0:
            gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
            print("[✓] Dataset successfully compiled!")
        else:
            print("No demonstrations to save. Exiting cleanly.")