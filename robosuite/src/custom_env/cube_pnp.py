import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

class CubePickAndPlace(ManipulationEnv):
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
            table_friction=(1., 5e-3, 1e-4),
            table_offset=(0, 0, 0.8),
        )

        # 2. Define the Box (Dynamic, pickable object)
        self.box_red = BoxObject(name="cube_red", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1])
        
        # 3. Define the Cylinders as "Ghost" Markers 
        self.cylinder_blue = CylinderObject(name="cylinder_blue", size=[0.06, 0.001], rgba=[0, 0, 1, 1], joints=None, obj_type="visual")
        self.cylinder_green = CylinderObject(name="cylinder_green", size=[0.06, 0.001], rgba=[0, 1, 0, 1], joints=None, obj_type="visual")

        # FUSE the cylinders directly into the Arena (Welds them to the world)
        for cyl in [self.cylinder_blue, self.cylinder_green]:
            self.arena.worldbody.append(cyl.get_obj())

        # 4. Compile the Task
        # ONLY pass the box to mujoco_objects. The cylinders are now permanently part of the table!
        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_red] 
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
        angles = [2 * np.pi / 3, 4 * np.pi / 3]
        cylinders = [self.cylinder_blue, self.cylinder_green]
        
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
        box_green_pos = positions[0].copy() # Green is the 1st cylinder
        box_green_pos[0] += np.random.randn() * 0.01
        box_green_pos[1] += np.random.randn() * 0.01

        # Place them flush on the table (Removed the +0.01cm drop height so they don't bounce)
        box_green_pos[2] = table_z + box_half_height
        
        default_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(self.box_red.joints[0], np.concatenate([box_green_pos, default_quat]))

        # CRITICAL: Move forward() to the VERY END so the physics engine doesn't explode!
        self.sim.forward()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Get body IDs for objects
        self.box_body_id = self.sim.model.body_name2id(self.box_red.root_body)
        self.cylinder_blue_body_id = self.sim.model.body_name2id(self.cylinder_blue.root_body)
        self.cylinder_green_body_id = self.sim.model.body_name2id(self.cylinder_green.root_body)
        
        # Store target position for blue cylinder (will be updated in _reset_internal)
        self.target_cylinder_pos = np.zeros(3)

    def reward(self, action=None):
        return 0 
            
    def _check_success(self):
        """
        Check if the box has been successfully placed near the center of the blue cylinder
        and the gripper is open (i.e., the box has been released).
        """
        # --- 1. Position check ---
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        target_pos = self.sim.data.body_xpos[self.cylinder_green_body_id]

        xy_dist = np.linalg.norm(box_pos[:2] - target_pos[:2])

        table_z = self.arena.table_top_abs[2]
        box_half_height = 0.02
        z_check = abs(box_pos[2] - (table_z + box_half_height)) < 0.02

        cylinder_radius = 0.06
        placed_correctly = xy_dist < cylinder_radius and z_check

        arm = self.robots[0].arms[0]
        # arm_gripper = arm.format_action()
        # print(f"Arm: {arm_gripper}")
        gripper_qpos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
        # print(f"Gripper QPOS: {gripper_qpos}")
        # current_qpos = self.robots[0].gripper[arm].format_action(gripper_qpos)
        # print(f"Current Gripper QPOS: {current_qpos}")


        # --- 2. Gripper open check ---
        # 1. Fetch the exact simulator indices for this robot's gripper joints
        gripper_joint_indices = self.robots[0]._ref_gripper_joint_pos_indexes
        # print(f"Gripper Joint Indices: {gripper_joint_indices}")
        
        # 2. Extract the current joint positions (angles/slider states) for the gripper
        # gripper_qpos = self.sim.data.qpos[7:9]
        gripper_qpos = self.sim.data.qpos[7]
        print(f"Gripper QPOS: {gripper_qpos}")
        
        # 3. Get the default fully open joint positions from the gripper model
        open_qpos = self.robots[0].gripper[arm].init_qpos[0]
        print(f"Open QPOS: {open_qpos}")
        # open_qpos = self.robots[0].gripper.init_qpos
        
        # 4. Check if current positions are very close to the open positions
        # A threshold of 0.01 or 0.02 is standard to account for minor physics jitter
        gripper_is_open = np.abs(gripper_qpos) - np.abs(open_qpos) > 0.01
        print(f"Gripper is open: {gripper_is_open}")

        success = placed_correctly and gripper_is_open
        
        return success