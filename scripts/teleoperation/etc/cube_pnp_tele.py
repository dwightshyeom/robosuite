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
from robosuite.environments.cube_pnp import CubePickAndPlace

# Main Data Collection Pipeline
if __name__ == "__main__":
    rclpy.init()
    node = TeleopNode()
    print("ROS 2 Started (Synchronous mode). Waiting for /arm/measured_cp...")

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")

    # ROBOSUITE CONFIGURATION DICTIONARY
    config = {
        "env_name": "PickAndPlace",
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    env_info_json = json.dumps(config)

    # ENVIRONMENT INITIALIZATION
    env = CubePickAndPlace(
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,            
        renderer='mjviewer',           
        has_offscreen_renderer=False,  
        use_camera_obs=False,          
        control_freq=20,
        ignore_done=True,              
    )

    env = VisualizationWrapper(env) # end-effector markers
    tmp_directory = f"/dev/shm/teleop_{str(time.time()).replace('.', '_')}"
    env = DataCollectionWrapper(env, tmp_directory)

    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)
    
    print("\n" + "="*50)
    print("INTERACTIVE DATA COLLECTION PIPELINE")
    print("="*50)

    num_successful_demos = 0

    try:
        # OUTER LOOP: Manages Episodes
        while True:
            user_input = input(f"\n[Dataset: {num_successful_demos} Demos] Press [ENTER] to start recording (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            print("Recording started! Move the robot to complete the task.")
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

                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
                env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

                obs, reward, done, info = env.step(env_action)
                env.render()
                
                # Success state machine
                if env._check_success():
                    success_hold_count -= 1
                else:
                    success_hold_count = 10 # maintain the requirement for 10 consecutive successful steps to trigger demo completion
                    
                if success_hold_count <= 0:
                    print("\n[✓] Task Objective Reached!")
                    break 
                
                diff = (1 / 20) - (time.time() - start_time)
                if diff > 0:
                    time.sleep(diff)

            # POST-DEMONSTRATION PROMPT
            save_input = input("Do you want to SAVE this demonstration? (y/n): ")
            
            if save_input.lower() == 'y':
                print("[+] Demonstration kept in temporary memory.")
                num_successful_demos += 1
            else:
                print("[-] Discarding data...")
                # Clear internal wrapper buffers
                env.states = []
                env.action_infos = []
                # Wipe from RAM
                if os.path.exists(env.ep_directory):
                    shutil.rmtree(env.ep_directory)
                    os.makedirs(env.ep_directory, exist_ok=True)
                print("    Data successfully wiped.")

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        rclpy.shutdown()
        env.close()

        print(f"\nCompiling {num_successful_demos} saved demonstrations into `.hdf5` dataset...")
        final_out_dir = "./custom_dataset"
        
        if num_successful_demos > 0:
            gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
        else:
            print("No demonstrations to save. Exiting cleanly.")