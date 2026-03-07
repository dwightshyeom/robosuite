import os
import time
import json
import h5py
import shutil
import datetime
import numpy as np
from glob import glob
from copy import deepcopy

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.controllers.composite.composite_controller import WholeBody

# from robosuite.src.utils.dataset_utils import gather_demonstrations_as_hdf5
from robosuite.src.device.phantom import PhantomOmni
from robosuite.src.custom_env import CubePickAndPlace
# ROS & Threading Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import transforms3d as tf3d

# --- 1. ROS 2 Node ---
class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_node")
        self.get_logger().info("TeleopNode has been initialized.")
        self.create_subscription(PoseStamped, "/arm/measured_cp", self.master_pose_callback, 10)
        
        self.master_position = None
        self.master_quat = None
        self.master_R = None

    def master_pose_callback(self, msg: PoseStamped):
        self.master_position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])
        self.master_quat = np.array([
            msg.pose.orientation.w, msg.pose.orientation.x, 
            msg.pose.orientation.y, msg.pose.orientation.z
        ])
        self.master_R = tf3d.quaternions.quat2mat(self.master_quat)


# --- 2. Data Compilation (Using your exact fixed version) ---
def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """Compiles temporary robosuite npz files into a robomimic-compatible hdf5 file."""
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None 

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states, actions = [], []
        success = False

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if not states: continue

        if success:
            del states[-1] 
            assert len(states) == len(actions)
            num_eps += 1
            ep_data_grp = grp.create_group(f"demo_{num_eps}")
            
            with open(os.path.join(directory, ep_directory, "model.xml"), "r") as xml_f:
                ep_data_grp.attrs["model_file"] = xml_f.read()

            ep_data_grp.create_dataset("states", data=np.array(states))
            ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
    grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name if env_name else "Lift"
    
    # ----------------------------------------------------
    # CRITICAL FIX: Saves natively as 'env_info' 
    # convert_robosuite.py will find this and parse it!
    # ----------------------------------------------------
    grp.attrs["env_info"] = env_info
    
    f.close()
    print(f"\nFinal dataset saved to: {hdf5_path}")


# --- 3. Main Data Collection Pipeline ---
if __name__ == "__main__":
    rclpy.init()
    node = TeleopNode()
    print("ROS 2 Started (Synchronous mode). Waiting for /arm/measured_cp...")

    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    
    # ----------------------------------------------------
    # THE NATIVE CONFIGURATION DICTIONARY
    # Flat structure matching the original Robosuite format
    # ----------------------------------------------------
    # config = {
    #     "env_name": "Lift",
    #     "robots": ["Panda"],
    #     "controller_configs": controller_config,
    # }
    config = {
        "env_name": "PickAndPlace",
        "robots": ["Panda"],
        "controller_configs": controller_config,
    }
    env_info_json = json.dumps(config)

    # ----------------------------------------------------
    # ENVIRONMENT INITIALIZATION
    # We unpack **config directly into the factory builder
    # ----------------------------------------------------
    # env = suite.make(
    #     **config,                       # Unpacks: env_name, robots, controller_configs
    #     has_renderer=True,             
    #     has_offscreen_renderer=True,    
    #     use_camera_obs=True,            
    #     camera_names="frontview",       
    #     camera_heights=512,             
    #     camera_widths=512,
    #     control_freq=20,
    #     ignore_done=True,               # We manually control episode resets
    # )

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

    env = VisualizationWrapper(env)
    tmp_directory = f"/dev/shm/teleop_{str(time.time()).replace('.', '_')}"
    env = DataCollectionWrapper(env, tmp_directory)

    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)
    
    print("\n" + "="*50)
    print("INTERACTIVE DATA COLLECTION PIPELINE")
    print("="*50)

    num_successful_demos = 0

    try:
        # --- OUTER LOOP: Manages Episodes ---
        while True:
            user_input = input(f"\n[Dataset: {num_successful_demos} Demos] Press [ENTER] to start recording (or type 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            
            print("Recording started! Move the robot to complete the task.")
            obs = env.reset()
            device.start_control()
            
            success_hold_count = 10 
            all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]

            # --- INNER LOOP: Teleoperation ---
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
                    success_hold_count = 10 
                    
                if success_hold_count <= 0:
                    print("\n[✓] Task Objective Reached!")
                    break 
                
                diff = (1 / 20) - (time.time() - start_time)
                if diff > 0:
                    time.sleep(diff)

            # --- POST-DEMONSTRATION PROMPT ---
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