import numpy as np
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.devices import Device
from robosuite.controllers.composite.composite_controller import WholeBody
import robosuite.utils.transform_utils as T

from robosuite.src.custom_env import FruitSwap

# ROS & Threading Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import threading
import transforms3d as tf3d

# External Imports
from pynput.keyboard import Key, Listener
import time
import os
import json
import h5py
import datetime
from glob import glob
from copy import deepcopy
import imageio

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
        # Commented out the print statement to keep the terminal clean during recording
        # self.get_logger().info("Received master pose...")
        self.master_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.master_quat = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z
        ])
        self.master_R = tf3d.quaternions.quat2mat(self.master_quat)

# --- 2. Custom Robosuite Device ---
class PhantomOmni(Device):
    """
    A custom device class that reads absolute poses from a ROS 2 node,
    calculates the delta movements, and packages them for Robosuite.
    (ORIENTATION CONTROL DISABLED for easier 3-DoF teleoperation)
    """
    def __init__(self, env, ros_node, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env)
        self.ros_node = ros_node
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity 
        
        self._reset_internal_state()

        # Keyboard listener now checks for BOTH press and release
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.last_pos = None
        self._grasp = False
        self._reset_state = 0
        self._enabled = False
        self._clutch_active = False # <--- Tracks if SHIFT is held down

    def start_control(self):
        self._reset_internal_state()
        self._enabled = True

    def get_controller_state(self):
        if self.ros_node.master_position is None or not self._enabled:
            return dict(dpos=np.zeros(3), rotation=np.eye(3), raw_drotation=np.zeros(3), grasp=False, reset=self._reset_state, base_mode=0)

        current_pos = self.ros_node.master_position.copy()

        if self.last_pos is None:
            mujoco_x = 2*current_pos[2]  
            mujoco_y = 2*current_pos[0]  
            mujoco_z = 1*current_pos[1]  
            self.last_pos = np.array([mujoco_x, mujoco_y, mujoco_z])

        # Apply your calibration mapping
        mujoco_x = 2*current_pos[2]  
        mujoco_y = 2*current_pos[0]  
        mujoco_z = 1*current_pos[1]  
        mapped_pos = np.array([mujoco_x, mujoco_y, mujoco_z])
        
        # Always calculate Delta to invisibly track the stylus
        dpos = mapped_pos - self.last_pos

        # Update baseline for the next frame
        self.last_pos = mapped_pos

        # CRITICAL CLUTCH LOGIC: If Shift is NOT held, wipe the movement delta to 0. 
        if not self._clutch_active:
            dpos = np.zeros(3)

        return dict(
            dpos=dpos * self.pos_sensitivity,
            rotation=np.eye(3),                       
            raw_drotation=np.zeros(3),                
            grasp=self._grasp,                        # <--- FIXED: Now passes a pure Boolean
            reset=self._reset_state,
            base_mode=0
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        dpos = dpos * 125
        dpos = np.clip(dpos, -1, 1)
        return dpos, drotation

    def on_press(self, key):
        # Engage clutch when Shift is pressed down
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = True
            print("\nClutch Engaged: You can now move the stylus without affecting the robot.\n")
        elif key == Key.esc:
            self._reset_state = 1
            self._enabled = False

    def on_release(self, key):
        # Disengage clutch when Shift is released
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = False
            
        # Toggle gripper on spacebar RELEASE to avoid key-repeat stuttering
        elif key == Key.space:
            self._grasp = not self._grasp
            print("\nToggling gripper...\n")

# --- 3. Custom Environment ---
class MultiObjectEnv(ManipulationEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        # 1. Define the Workspace
        table_full_size = (0.8, 0.8, 0.05)
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        self.arena = TableArena(
            table_full_size=table_full_size,
            # table_friction=(10000., 1000., 1000.),
            table_friction=(1., 5e-3, 1e-4),
            table_offset=(0, 0, 0.8),
        )

        # 2. Define the Box (Dynamic, pickable object)
        self.box_red = BoxObject(name="cube_red", size=[0.02, 0.02, 0.02], rgba=[1, 0, 1, 1])
        self.box_green = BoxObject(name="cube_green", size=[0.02, 0.02, 0.02], rgba=[0, 1, 1, 1])
        
        # 3. Define the Cylinders as "Ghost" Markers 
        # obj_type="visual" removes all physical collision boundaries.
        self.cylinder_red = CylinderObject(name="cylinder_red", size=[0.06, 0.001], rgba=[1, 0, 0, 1], joints=None, obj_type="visual")
        self.cylinder_blue = CylinderObject(name="cylinder_blue", size=[0.06, 0.001], rgba=[0, 0, 1, 1], joints=None, obj_type="visual")
        self.cylinder_green = CylinderObject(name="cylinder_green", size=[0.06, 0.001], rgba=[0, 1, 0, 1], joints=None, obj_type="visual")

        # FUSE the cylinders directly into the Arena (Welds them to the world)
        for cyl in [self.cylinder_red, self.cylinder_blue, self.cylinder_green]:
            self.arena.worldbody.append(cyl.get_obj())

        # 4. Compile the Task
        # ONLY pass the box to mujoco_objects. The cylinders are now permanently part of the table!
        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_red, self.box_green] 
        )

    def _reset_internal(self):
        super()._reset_internal()
        
        # 1. Set a FIXED center point (Removed np.random so it stays perfectly still!)
        center_x = 0.0
        center_y = 0.0
        table_z = self.arena.table_top_abs[2]
        
        cyl_half_height = 0.001
        cyl_z = table_z + cyl_half_height

        # 2. Calculate coordinates (120 degrees apart)
        radius = 0.12
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        cylinders = [self.cylinder_red, self.cylinder_blue, self.cylinder_green]
        
        positions = []
        
        for angle, cyl in zip(angles, cylinders):
            pos_x = center_x + radius * np.cos(angle)
            pos_y = center_y + radius * np.sin(angle)
            cyl_pos = np.array([pos_x, pos_y, cyl_z])
            positions.append(cyl_pos)
            
            # Move the static visual markers directly in the physics engine
            body_id = self.sim.model.body_name2id(cyl.root_body)
            self.sim.model.body_pos[body_id] = cyl_pos

        # 3. Place the boxes exactly on the red and green cylinders
        box_half_height = 0.02
        box_red_pos = positions[0].copy()   # Red is the 1st cylinder
        box_green_pos = positions[2].copy() # Green is the 3rd cylinder
        
        # Place them flush on the table (Removed the +0.01cm drop height so they don't bounce)
        box_red_pos[2] = table_z + box_half_height
        box_green_pos[2] = table_z + box_half_height
        
        default_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(self.box_red.joints[0], np.concatenate([box_red_pos, default_quat]))
        self.sim.data.set_joint_qpos(self.box_green.joints[0], np.concatenate([box_green_pos, default_quat]))

        # CRITICAL: Move forward() to the VERY END so the physics engine doesn't explode!
        self.sim.forward()

    def reward(self, action=None):
        return 0 
        
    def _check_success(self):
        # DataCollectionWrapper only saves runs that return True here.
        # Hardcoded to True for data collection testing.
        return True 


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
    grp.attrs["env"] = env_name if env_name else "MultiObjectEnv"
    grp.attrs["env_args"] = env_info
    f.close()
    print(f"\nFinal dataset saved to: {hdf5_path}")


# if __name__ == "__main__":
#     # --- Boot up ROS 2 ---
#     rclpy.init()
#     node = TeleopNode()
#     # executor = rclpy.executors.MultiThreadedExecutor()
#     # executor.add_node(node)
#     # ros_thread = threading.Thread(target=executor.spin, daemon=True)
#     # ros_thread.start()
#     print("ROS 2 Thread Started. Waiting for /arm/measured_cp...")

#     # --- Boot up Robosuite ---
#     controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    
#     config_dict = {
#         "env_name": "MultiObjectEnv",
#         "type": 1,  # 1 tells robomimic this is a robosuite-type environment
#         "env_kwargs": {
#             "robots": ["Panda"],
#             "controller_configs": controller_config,
#             "has_renderer": False,
#             "has_offscreen_renderer": False,
#             "use_camera_obs": False,
#         }
#     }

#     env_info_json = json.dumps(config_dict)

#     env = FruitSwap(
#         robots="Panda", 
#         controller_configs=controller_config,
#         has_renderer=True,            
#         renderer='mujoco',
#         # renderer="mjviewer",           # <--- CRITICAL FIX: GPU-Accelerated Viewer
#         has_offscreen_renderer=False,  
#         use_camera_obs=False,          
#         control_freq=20,
#         ignore_done=True,
#     )
    
#     env = VisualizationWrapper(env)
#     tmp_directory = f"/tmp/teleop_{str(time.time()).replace('.', '_')}"
#     env = DataCollectionWrapper(env, tmp_directory)

#     # Initialize our custom Phantom Omni Device
#     device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)
    
#     obs = env.reset()
#     env.render()
#     device.start_control()
    
#     print("Environment loaded!")
#     print("Use your Phantom Omni to move. Press SPACE to toggle gripper. Press ESC to stop.")

#     all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]

#     try:
        
#         while True:
#             start_time = time.time()
#             rclpy.spin_once(node, timeout_sec=0.0)
#             active_robot = env.robots[device.active_robot]
            
#             # The base class automatically calls our custom get_controller_state()
#             input_ac_dict = device.input2action()
            
#             if input_ac_dict is None:
#                 break 
                
#             action_dict = deepcopy(input_ac_dict)
            
#             for arm in active_robot.arms:
#                 if isinstance(active_robot.composite_controller, WholeBody): 
#                     controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
#                 else:
#                     controller_input_type = active_robot.part_controllers[arm].input_type

#                 if controller_input_type == "delta":
#                     action_dict[arm] = input_ac_dict[f"{arm}_delta"]
#                 elif controller_input_type == "absolute":
#                     action_dict[arm] = input_ac_dict[f"{arm}_abs"]

#             env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(env.robots)]
#             env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
#             env_action = np.concatenate(env_action)
            
#             for gripper_ac in all_prev_gripper_actions[device.active_robot]:
#                 all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

#             obs, reward, done, info = env.step(env_action)
#             # env.render()
            
#             diff = (1 / 20) - (time.time() - start_time)
#             if diff > 0:
#                 time.sleep(diff)

#     except KeyboardInterrupt:
#         pass
#     finally:
#         rclpy.shutdown()
#         env.close()

#         print("\nCompiling `.npz` files into `.hdf5` dataset...")
#         final_out_dir = "./custom_dataset"
#         gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)

if __name__ == "__main__":
    # --- Boot up ROS 2 ---
    rclpy.init()
    node = TeleopNode()
    print("ROS 2 Started (Synchronous mode). Waiting for /arm/measured_cp...")

    # --- Boot up Robosuite ---
    controller_config = load_composite_controller_config(controller="BASIC", robot="Panda")
    
    config_dict = {
        "env_name": "MultiObjectEnv",
        "type": 1,
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
        }
    }

    env_info_json = json.dumps(config_dict)

    env = FruitSwap(
        robots="Panda", 
        controller_configs=controller_config,
        has_renderer=True,            
        renderer='mujoco',             # CPU Software Renderer
        has_offscreen_renderer=True,   # <--- TURNED ON to generate video frames
        use_camera_obs=True,           # <--- TURNED ON to pull RGB arrays
        camera_names="frontview",      # <--- Specifies the camera angle
        camera_heights=512,            # <--- Video resolution height
        camera_widths=512,             # <--- Video resolution width
        control_freq=20,
        ignore_done=True,
    )
    
    env = VisualizationWrapper(env)
    tmp_directory = f"/tmp/teleop_{str(time.time()).replace('.', '_')}"
    env = DataCollectionWrapper(env, tmp_directory)

    # Initialize our custom Phantom Omni Device
    device = PhantomOmni(env=env, ros_node=node, pos_sensitivity=1.0, rot_sensitivity=1.0)
    
    obs = env.reset()
    env.render()
    device.start_control()
    
    print("Environment loaded!")
    print("Use your Phantom Omni to move. Press SPACE to toggle gripper. Press ESC to stop.")

    all_prev_gripper_actions = [{ f"{arm}_gripper": np.zeros(robot.gripper[arm].dof) for arm in robot.arms if robot.gripper[arm].dof > 0 } for robot in env.robots]

    # --- Initialize Video Writer ---
    video_path = "live_teleop_recording.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)
    print(f"Recording live video to {video_path}...")

    try:
        # Capture the very first frame
        initial_frame = np.flipud(obs["frontview_image"])
        video_writer.append_data(initial_frame)
        
        while True:
            start_time = time.time()
            rclpy.spin_once(node, timeout_sec=0.0)
            active_robot = env.robots[device.active_robot]
            
            input_ac_dict = device.input2action()
            
            if input_ac_dict is None:
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
            env.render() # Renders to the OpenCV window
            
            # --- Capture and save the video frame ---
            frame = np.flipud(obs["frontview_image"])
            video_writer.append_data(frame)
            
            diff = (1 / 20) - (time.time() - start_time)
            if diff > 0:
                time.sleep(diff)

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        env.close()

        # --- Safely close the video writer ---
        video_writer.close()
        print(f"\nLive video successfully saved to {video_path}")

        print("\nCompiling `.npz` files into `.hdf5` dataset...")
        final_out_dir = "./custom_dataset"
        gather_demonstrations_as_hdf5(tmp_directory, final_out_dir, env_info_json)
