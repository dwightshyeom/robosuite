import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

class CubePickAndPlace(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )



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

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment.
        These define the "object" array that robomimic uses for training.
        """
        # 1. Get baseline observables from parent class (robot states, eef, etc.)
        observables = super()._setup_observables()

        # Get prefix from robot model to access the end-effector position dynamically
        pf = self.robots[0].robot_model.naming_prefix

        # --- 2. Sensors for the Pickable Box ---
        @sensor(modality="object")
        def cube_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.box_body_id])

        @sensor(modality="object")
        def cube_quat(obs_cache):
            return T.convert_quat(self.sim.data.body_xquat[self.box_body_id], to="xyzw")

        @sensor(modality="object")
        def gripper_to_cube_pos(obs_cache):
            # Calculating distance from gripper to cube helps the neural network learn faster
            return obs_cache["cube_pos"] - obs_cache[f"{pf}eef_pos"] if \
                "cube_pos" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

        # --- 3. Sensors for the Target Goals (Cylinders) ---
        @sensor(modality="object")
        def cylinder_blue_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cylinder_blue_body_id])

        @sensor(modality="object")
        def cylinder_green_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.cylinder_green_body_id])

        # 4. Compile all sensors
        sensors = [cube_pos, cube_quat, gripper_to_cube_pos, cylinder_blue_pos, cylinder_green_pos]
        names = ["cube_pos", "cube_quat", "gripper_to_cube_pos", "cylinder_blue_pos", "cylinder_green_pos"]

        # 5. Register them to the environment
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

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
        gripper_qpos = self.sim.data.qpos[8]
        open_qpos = self.robots[0].gripper[arm].init_qpos[0] * 2
        gripper_is_open = np.abs(np.abs(open_qpos) - np.abs(gripper_qpos)) < 0.005
        success = placed_correctly and gripper_is_open
        
        return success