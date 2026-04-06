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
from robosuite.environments.pomdp.fruit_swap_vision import FruitSwapVision
from robosuite.src.utils.dataset_utils import gather_demonstrations_as_hdf5
from robosuite.src.utils.teleop_node_utils import TeleopNode
from robosuite.src.device.phantom import PhantomOmni

if __name__ == "__main__":
    rclpy.init()
    node = TeleopNode()
    print("ROS 2 Started. Waiting for /arm/measured_cp...")

    while node.master_position is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("[!] Phantom Omni connected.")

    # Load standard controller configurations
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

    config = {
        "env_name": "FruitSwapVision",
        "robots": ["Panda"], # Assuming bimanual setup for swapping
        "controller_configs": controller_config,
    }
    env_info_json = json.dumps(config)

    # Initialize the custom environment
    env = FruitSwapVision(
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,            
        renderer='mjviewer',           
        has_offscreen_renderer=True,  
        use_camera_obs=False,          
        control_freq=20,
        ignore_done=True,     
    )

    # tmp_directory = "/tmp/fruit_swap_teleop_data"
    # if os.path.exists(tmp_directory):
    #     shutil.rmtree(tmp_directory)
        
    env = VisualizationWrapper(env)
    tmp_directory = f"/dev/shm/teleop_{str(time.time()).replace('.', '_')}"
    env = DataCollectionWrapper(env, tmp_directory)

    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)

    num_successful_demos = 0
    num_flipped_demos = 0

    print("\n" + "="*50)
    print(" FRUIT SWAP TELEOPERATION STARTED ")
    print("="*50)

    try:
        while True:
            print(f"Saved: {num_successful_demos} total, {num_flipped_demos} flipped, {num_successful_demos - num_flipped_demos} normal")
            user_input = input("Press [ENTER] to start recording (or type 'q' to quit): ")
            
            if user_input.lower() == 'q':
                break

            print(f"\n[Episode {num_successful_demos + 1}] Ready. Press the Omni button to start moving...")
            
            print(">>> Recording Started! <<<")
            step_count = 0

            obs = env.reset()
            device.start_control()

            success_hold_count = 10
            all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]
            
            while True:
                start_time = time.time()
                rclpy.spin_once(node, timeout_sec=0.001)

                active_robot = env.robots[device.active_robot]
                
                input_ac_dict = device.input2action()
                
                # ESC manually aborts the loop
                if input_ac_dict is None:
                    print("\n[!] ESC pressed. Halting current demonstration.")
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
                # step_count += 1
                
                # =======================================================
                # AUTOMATIC SUCCESS CHECK
                # =======================================================
                # Success state machine (Automatically uses the target_dist we set!)
                if env._check_success():
                    print(f"[✓] Success condition met! Hold for {success_hold_count} more steps...")
                    success_hold_count -= 1
                else:
                    success_hold_count = 10 
                    
                # Auto-break when task is completed
                if success_hold_count <= 0:
                    print(f"\n[✓] Goal Reached!")
                    break 

                diff = (1 / 20) - (time.time() - start_time)
                if diff > 0:
                    time.sleep(diff)

            # --- POST-DEMONSTRATION PROMPT ---
            was_flipped = env.flip
            
            print(f"    Flip: {was_flipped}")
            save_input = input("\nDo you want to SAVE this demonstration? (y/n): ")
            
            if save_input.lower() == 'y':
                print("[+] Demonstration kept in temporary memory.")
                num_successful_demos += 1
                if was_flipped:
                    num_flipped_demos += 1
            else:
                print("[-] Discarding data...")
                # Clear the data collection wrapper's internal memory
                env.states = []
                env.action_infos = []
                # Wipe the specific episode folder from the tmp directory
                if hasattr(env, 'ep_directory') and os.path.exists(env.ep_directory):
                    shutil.rmtree(env.ep_directory)
                    os.makedirs(env.ep_directory, exist_ok=True)
                print("    Data successfully wiped.")

    except KeyboardInterrupt:
        print("\n[!] Pipeline interrupted by user.")
    finally:
        rclpy.shutdown()
        env.close()

        print(f"\nCompiling {num_successful_demos} saved demonstrations into `.hdf5` dataset...")
        print(f"  Flipped: {num_flipped_demos} | Normal: {num_successful_demos - num_flipped_demos}")
        final_out_dir = "./custom_dataset" 
        
        if num_successful_demos > 0:
            gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
            print("[✓] Dataset successfully compiled!")
        else:
            print("No demonstrations to save. Exiting cleanly.")