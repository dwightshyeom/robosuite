import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

class FruitSwap(ManipulationEnv):
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
        return False