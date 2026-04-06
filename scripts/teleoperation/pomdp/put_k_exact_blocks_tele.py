import os
import time
import json
import rclpy
import shutil
import numpy as np
import glob
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

# ============================================
# Import the task environment
# ============================================
from robosuite.environments.pomdp.put_k_exact_blocks import PutKExactBlocks

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
        "env_name": "PutKExactBlocks",
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    env_info_json = json.dumps(config)

    # ============================================
    # Set task environment configuration
    # ============================================
    env = PutKExactBlocks(
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,            
        renderer='mjviewer',           
        has_offscreen_renderer=False,  
        use_camera_obs=False,          
        control_freq=20,
        ignore_done=True, 
    )
    
    # Wrap environment for data collection and visualization
    tmp_directory = f"/dev/shm/teleop_{str(time.time()).replace('.', '_')}"
    data_wrapper = DataCollectionWrapper(env, tmp_directory)
    env = VisualizationWrapper(data_wrapper)

    # Initialize the Phantom Omni device
    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)
 
    num_successful_demos = 0
    
    print("\n" + "="*50)
    print("  DATA COLLECTION PIPELINE")
    print("  Use your Phantom Omni to move. Press SPACE to toggle gripper. Press ESC to stop.")
    print("="*50)

    try:
        while True:
            print(f"Saved: {num_successful_demos} total demonstrations.")
            user_input = input("Press [ENTER] to start recording (or type 'q' to quit): ")
            
            if user_input.lower() == 'q':
                break

            print(f"\n[Episode {num_successful_demos + 1}] Ready. Press the Omni button to start moving...")
            
            print(">>> Recording Started! <<<")
            step_count = 0

            obs = env.reset()
            device.start_control()

            success_hold_count = 10
            failure_flag = False  # Track if the episode failed
            all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]
            
            while True:
                start_time = time.time()
                rclpy.spin_once(node, timeout_sec=0.001)

                active_robot = env.robots[device.active_robot]
                
                input_ac_dict = device.input2action()
                
                # ESC manually aborts the loop
                if input_ac_dict is None:
                    print("\n[!] ESC pressed. Halting current demonstration.")
                    failure_flag = True
                    break 
                    
                action_dict = deepcopy(input_ac_dict)

                for arm in active_robot.arms:
                    if isinstance(active_robot.composite_controller, WholeBody): 
                        controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = active_robot.part_controllers[arm].input_type

                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]

                # Construct the full environment action vector across all robots
                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
                env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                
                # Update the stored previous actions
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

                obs, reward, done, info = env.step(env_action)
                env.render() 
                
                # --- NEW STATE MACHINE LOGIC ---
                
                # 1. Instantly check for explicit failure
                if env._check_failure():
                    print("\n[X] FAILURE CONDITION MET!")
                    print("    Reason: Pressed button early, overflowed the box, or dropped the block.")
                    failure_flag = True
                    break

                # 2. Check for success
                if env._check_success():
                    print(f"[✓] Success condition met! Hold for {success_hold_count} more steps...")
                    success_hold_count -= 1
                else:
                    success_hold_count = 10 
                    
                # 3. Auto-break when task is completed perfectly
                if success_hold_count <= 0:
                    print(f"\n[✓] Goal Reached!")
                    failure_flag = False
                    break 

                diff = (1 / 20) - (time.time() - start_time)
                if diff > 0:
                    time.sleep(diff)

            # --- AUTOMATIC DISCARD LOGIC ---
            if failure_flag:
                print("[-] Automatically discarding failed demonstration...")
                # Clear the data collection wrapper's internal memory
                env.states = []
                env.action_infos = []
                # Wipe the specific episode folder from the tmp directory
                if hasattr(env, 'ep_directory') and os.path.exists(env.ep_directory):
                    shutil.rmtree(env.ep_directory)
                    os.makedirs(env.ep_directory, exist_ok=True)
                print("    Data successfully wiped. Please try again.")
            
            else:
                # Only ask to save if the trajectory was a true success
                save_input = input("\nDo you want to SAVE this demonstration? (y/n): ")
                
                if save_input.lower() == 'y':
                    print("[+] Demonstration kept in temporary memory.")
                    num_successful_demos += 1
                else:
                    print("[-] Discarding data...")
                    env.states = []
                    env.action_infos = []
                    if hasattr(env, 'ep_directory') and os.path.exists(env.ep_directory):
                        shutil.rmtree(env.ep_directory)
                        os.makedirs(env.ep_directory, exist_ok=True)
                    print("    Data successfully wiped.")

    except KeyboardInterrupt:
        print("\n[!] Pipeline interrupted by user.")
    finally:
        rclpy.shutdown()
        env.close()

        ep_folders = glob.glob(os.path.join(tmp_directory, "ep_*"))
        for ep_dir in ep_folders:
            condition_file = os.path.join(ep_dir, "model.xml")
            if not os.path.exists(condition_file):
                shutil.rmtree(ep_dir) # Vaporize the ghost folder

        print(f"\nCompiling {num_successful_demos} saved demonstrations into `.hdf5` dataset...")
        final_out_dir = "./custom_dataset" 
        
        if num_successful_demos > 0:
            gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
            print("[✓] Dataset successfully compiled!")
        else:
            print("No demonstrations to save. Exiting cleanly.")
