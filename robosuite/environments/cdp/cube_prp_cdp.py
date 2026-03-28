import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

class CubePickRotatePlaceCDP(ManipulationEnv):
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
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        
        # The Circular Setup Parameters
        self.target_angle = 60.0 # Degrees (Default condition)
        self.circle_radius = 0.15 # Distance from center point (15cm)
        self.center_x = 0.0 
        self.center_y = 0.0

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

    # Pipeline Setter Method
    def set_target_angle(self, angle_deg):
        """Called by the data collection loop before env.reset() to inject condition"""
        self.target_angle = float(angle_deg)

    def _load_model(self):
        super()._load_model()

        # Robot base position
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        self.arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=(0, 0, 0.8),
        )

        # The Box to be picked and placed
        self.box_red = BoxObject(
            name="cube_red", 
            size=[0.02, 0.02, 0.02],  
            rgba=[1, 0, 0, 1]
        )
        
        # 1. Start Cylinder (Blue) at 0 degrees
        self.cyl_start = CylinderObject(name="cyl_0", size=[0.04, 0.001], rgba=[0, 0, 0, 0.6], joints=None, obj_type="visual")
        
        # 2. Target Cylinders (Green) at 60, 120, 180 degrees
        self.cyl_60 = CylinderObject(name="cyl_60", size=[0.04, 0.001], rgba=[0, 0, 1, 0.6], joints=None, obj_type="visual")
        self.cyl_120 = CylinderObject(name="cyl_120", size=[0.04, 0.001], rgba=[0, 1, 0, 0.6], joints=None, obj_type="visual")
        self.cyl_180 = CylinderObject(name="cyl_180", size=[0.04, 0.001], rgba=[1, 0, 0, 0.6], joints=None, obj_type="visual")

        for cyl in [self.cyl_start, self.cyl_60, self.cyl_120, self.cyl_180]:
            self.arena.worldbody.append(cyl.get_obj())

        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_red] 
        )

    def _get_circular_pos(self, angle_deg, z_height):
        """Helper to compute X, Y, Z based on angle and radius"""
        rad = np.deg2rad(angle_deg)
        x = self.center_x + self.circle_radius * np.cos(rad)
        y = self.center_y + self.circle_radius * np.sin(rad)
        return np.array([x, y, z_height])

    def _reset_internal(self):
        super()._reset_internal()
        table_z = self.arena.table_top_abs[2]

        # 1. Place the 4 visual cylinders in a semi-circle
        angles = [90, 30, 330, 270]
        cylinders = [self.cyl_start, self.cyl_60, self.cyl_120, self.cyl_180]
        
        for angle, cyl in zip(angles, cylinders):
            cyl_pos = self._get_circular_pos(angle, table_z + 0.001)
            body_id = self.sim.model.body_name2id(cyl.root_body)
            self.sim.model.body_pos[body_id] = cyl_pos

        # 2. Place the box on the 0-degree cylinder (Blue Start Cylinder)
        box_half_height = 0.02
        box_pos = self._get_circular_pos(90, table_z + box_half_height)
        
        # Add slight positional noise so it doesn't just memorize the exact coordinate
        box_pos[0] += np.random.uniform(-0.01, 0.01)
        box_pos[1] += np.random.uniform(-0.01, 0.01)
        
        default_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(self.box_red.joints[0], np.concatenate([box_pos, default_quat]))
        self.sim.forward()

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box_red.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        # Combine position and quaternion
        @sensor(modality="object")
        def object(obs_cache):
            pos = self.sim.data.body_xpos[self.box_body_id]
            quat = T.convert_quat(self.sim.data.body_xquat[self.box_body_id], to="xyzw")
            return np.concatenate([pos, quat])

        @sensor(modality="object")
        def gripper_to_box_pos(obs_cache):
            box_pos = self.sim.data.body_xpos[self.box_body_id]
            return box_pos - obs_cache[f"{pf}eef_pos"] if f"{pf}eef_pos" in obs_cache else np.zeros(3)

        # The conditioning parameter (Outputs the requested target angle: 60, 120, or 180)
        @sensor(modality="goal")
        def condition(obs_cache):
            return np.array([self.target_angle], dtype=np.float32)

        sensors = [object, gripper_to_box_pos, condition]
        names = ["object", "gripper_to_box_pos", "condition"]

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
        Check if the box has been placed on the correct physical target cylinder 
        based on the logical condition (self.target_angle), and the gripper is released.
        """
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        
        # 1. Map the logical condition to the actual physical cylinder
        if self.target_angle == 60.0:
            target_cyl = self.cyl_60
        elif self.target_angle == 120.0:
            target_cyl = self.cyl_120
        elif self.target_angle == 180.0:
            target_cyl = self.cyl_180
        else:
            return False # Invalid condition
            
        # Get the real-time physical position of the chosen cylinder
        target_cyl_id = self.sim.model.body_name2id(target_cyl.root_body)
        target_pos = self.sim.data.body_xpos[target_cyl_id]
        
        # 2. Position Check (Must be on top of the physical target cylinder)
        xy_dist_from_target = np.linalg.norm(box_pos[:2] - target_pos[:2])
        
        table_z = self.arena.table_top_abs[2]
        box_half_height = 0.02
        z_check = abs(box_pos[2] - (table_z + box_half_height)) < 0.01
        
        placed_correctly = (xy_dist_from_target < 0.04) and z_check
        
        # 3. Gripper Check (Must be released)
        arm = self.robots[0].arms[0]
        gripper_qpos = self.sim.data.qpos[8]
        open_qpos = self.robots[0].gripper[arm].init_qpos[0] * 2
        gripper_is_open = np.abs(np.abs(open_qpos) - np.abs(gripper_qpos)) < 0.005

        if gripper_is_open and placed_correctly:
            print(f'[✓] Goal Reached! Placed on {self.target_angle} Condition Cylinder | Distance Error: {xy_dist_from_target:.4f}m')
            return True
            
        return False