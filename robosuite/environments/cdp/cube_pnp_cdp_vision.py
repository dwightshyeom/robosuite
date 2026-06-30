import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask

from robosuite.utils.observables import Observable, sensor
import robosuite.utils.transform_utils as T

class CubePlaceCDPVision(ManipulationEnv):
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
        camera_names=("agentview", "robot0_eye_in_hand"), 
        camera_heights=84, 
        camera_widths=84,
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
        
        # The default condition (Will be overwritten by teleop pipeline)
        self.target_place_distance = 0.1 
        self.start_x = 0.0 
        self.start_y = 0.0

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

    # Pipeline Setter Method (Same logic as BoxPush)
    def set_target_distance(self, distance):
        """Called by the data collection loop before env.reset()"""
        self.target_place_distance = distance

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
            size=[0.02, 0.02, 0.02],  # Smaller 2cm box for picking
            rgba=[1, 0, 0, 1]
        )
        
        # Visual indicator bars (Thin along X-axis, wide along Y-axis to mark placement zones)
        self.bar_05 = BoxObject(name="bar_05", size=[0.35, 0.005, 0.001], rgba=[0, 1, 0, 0.4], joints=None, obj_type="visual")
        self.bar_10 = BoxObject(name="bar_10", size=[0.35, 0.005, 0.001], rgba=[0, 0, 1, 0.4], joints=None, obj_type="visual")
        self.bar_15 = BoxObject(name="bar_15", size=[0.35, 0.005, 0.001], rgba=[1, 1, 0, 0.4], joints=None, obj_type="visual")

        for bar in [self.bar_05, self.bar_10, self.bar_15]:
            self.arena.worldbody.append(bar.get_obj())

        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_red] 
        )

    def _reset_internal(self):
        super()._reset_internal()
        table_z = self.arena.table_top_abs[2]

        # Place the visual indicator bars along the X-axis
        distances = [0.1, 0.2, 0.3]
        bars = [self.bar_05, self.bar_10, self.bar_15]
        
        for bar, dist in zip(bars, distances):
            pos_y = self.start_y + dist
            bar_pos = np.array([0.0, pos_y, table_z + 0.001])
            body_id = self.sim.model.body_name2id(bar.root_body)
            self.sim.model.body_pos[body_id] = bar_pos

        # Place the box at the fixed starting location (0, 0)
        box_half_height = 0.02
        box_pos = np.array([self.start_x, self.start_y, table_z + box_half_height])
        
        # Add slight initialization noise so the policy learns robust grasping
        box_pos[0] += np.random.uniform(-0.005, 0.005)
        box_pos[1] += np.random.uniform(-0.005, 0.005)
        
        default_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(self.box_red.joints[0], np.concatenate([box_pos, default_quat]))
        self.sim.forward()

    def _setup_references(self):
        super()._setup_references()
        self.box_body_id = self.sim.model.body_name2id(self.box_red.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        # Combine position and quaternion to avoid the "object doesn't exist" error later
        @sensor(modality="object")
        def object(obs_cache):
            pos = self.sim.data.body_xpos[self.box_body_id]
            quat = T.convert_quat(self.sim.data.body_xquat[self.box_body_id], to="xyzw")
            return np.concatenate([pos, quat])

        @sensor(modality="object")
        def gripper_to_box_pos(obs_cache):
            box_pos = self.sim.data.body_xpos[self.box_body_id]
            return box_pos - obs_cache[f"{pf}eef_pos"] if f"{pf}eef_pos" in obs_cache else np.zeros(3)

        # The conditioning parameter
        @sensor(modality="goal")
        def place_distance(obs_cache):
            return np.array([self.target_place_distance], dtype=np.float32)

        sensors = [object, gripper_to_box_pos, place_distance]
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
        Check if the box has been successfully placed at the target X-distance,
        is resting on the table, and the gripper is open.
        """
        target_y = self.start_y + self.target_place_distance
        box_pos = self.sim.data.body_xpos[self.box_body_id]
        y_dist_to_target = abs(box_pos[1] - target_y)
        x_dist_from_center = abs(box_pos[0] - self.start_x)
        
        # Tolerance
        reached_distance = y_dist_to_target < 0.01 # Critical tolerance
        stayed_on_path = x_dist_from_center < 0.05
        
        placed_correctly = reached_distance and stayed_on_path
        
        # 3. Gripper Check (Must be released) - Exact code preserved from your cube_pnp.py
        arm = self.robots[0].arms[0]
        gripper_qpos = self.sim.data.qpos[8]
        open_qpos = self.robots[0].gripper[arm].init_qpos[0] * 2
        gripper_is_open = np.abs(np.abs(open_qpos) - np.abs(gripper_qpos)) < 0.005

        if placed_correctly and gripper_is_open:
            print(f'[✓] Distance from the target: {y_dist_to_target:.4f}m | Gripper is open.')
        return placed_correctly and gripper_is_open