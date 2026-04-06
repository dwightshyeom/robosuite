import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
    

class FruitSwapVision(ManipulationEnv):
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
        camera_heights=256, 
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.flip = False  # will be properly randomized on first real reset

        # # reward configuration
        # self.reward_scale = reward_scale
        # self.reward_shaping = reward_shaping

        # # whether to use ground-truth object states
        # self.use_object_obs = use_object_obs

        # # object placement initializer
        # self.placement_initializer = placement_initializer

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

        # 2. Define the Boxes
        self.box_magenta = BoxObject(name="cube_magenta", size=[0.02, 0.02, 0.02], rgba=[1, 0, 1, 1])
        self.box_cyan = BoxObject(name="cube_cyan", size=[0.02, 0.02, 0.02], rgba=[0, 1, 1, 1])

        # 3. Define the Cylinders as "Ghost" Markers
        self.cylinder_red = CylinderObject(name="cylinder_red", size=[0.06, 0.001], rgba=[1, 0, 0, 1], joints=None, obj_type="visual")
        self.cylinder_blue = CylinderObject(name="cylinder_blue", size=[0.06, 0.001], rgba=[0, 0, 1, 1], joints=None, obj_type="visual")
        self.cylinder_green = CylinderObject(name="cylinder_green", size=[0.06, 0.001], rgba=[0, 1, 0, 1], joints=None, obj_type="visual")

        # Weld cylinders to the table
        for cyl in [self.cylinder_red, self.cylinder_blue, self.cylinder_green]:
            self.arena.worldbody.append(cyl.get_obj())

        # 4. Compile the Task
        self.model = ManipulationTask(
            mujoco_arena=self.arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.box_magenta, self.box_cyan] 
        )

    def _setup_references(self):
        super()._setup_references()
        self.box_magenta_body_id = self.sim.model.body_name2id(self.box_magenta.root_body)
        self.box_cyan_body_id = self.sim.model.body_name2id(self.box_cyan.root_body)

        self.cylinder_red_body_id = self.sim.model.body_name2id(self.cylinder_red.root_body)
        self.cylinder_blue_body_id = self.sim.model.body_name2id(self.cylinder_blue.root_body)
        self.cylinder_green_body_id = self.sim.model.body_name2id(self.cylinder_green.root_body)

    def reward(self, action=None):
        return 0 

    def _reset_internal(self):
        super()._reset_internal()
        
        # 1. Set FIXED cylinder coordinates
        center_x, center_y = 0.0, 0.0
        table_z = self.arena.table_top_abs[2]
        cyl_z = table_z + 0.001
        radius = 0.12
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        
        cylinders = [self.cylinder_red, self.cylinder_blue, self.cylinder_green]
        positions = []
        
        for angle, cyl in zip(angles, cylinders):
            pos_x = center_x + radius * np.cos(angle)
            pos_y = center_y + radius * np.sin(angle)
            cyl_pos = np.array([pos_x, pos_y, cyl_z])
            positions.append(cyl_pos)
            
            body_id = self.sim.model.body_name2id(cyl.root_body)
            self.sim.model.body_pos[body_id] = cyl_pos

        self.pos_red   = positions[0].copy()
        self.pos_blue  = positions[1].copy()
        self.pos_green = positions[2].copy()

        # 2. Determine Flip and Starting Positions
        # flip=False: magenta on blue, cyan on green
        # flip=True:  magenta on green, cyan on blue
        # Only re-randomize flip on a true new episode, not on deterministic reloads
        if not self.deterministic_reset:
            self.flip = np.random.rand() < 0.5
        box_half_height = 0.02

        if not self.flip:
            self.magenta_start_pos = self.pos_blue.copy()
            self.cyan_start_pos    = self.pos_green.copy()
        else:
            self.magenta_start_pos = self.pos_green.copy()
            self.cyan_start_pos    = self.pos_blue.copy()

        # Phase 2: cyan goes where magenta started
        # Phase 3: magenta goes where cyan started
        self.cyan_target_pos   = self.magenta_start_pos.copy()
        self.magenta_final_pos = self.cyan_start_pos.copy()

        # Spawn positions with noise (separate arrays, never touch the target positions)
        start_magenta = self.magenta_start_pos.copy()
        start_cyan    = self.cyan_start_pos.copy()

        # print(f"[SPAWN DEBUG] flip={self.flip}")
        # print(f"[SPAWN DEBUG] pos_red={np.round(self.pos_red[:2],4)}  pos_blue={np.round(self.pos_blue[:2],4)}  pos_green={np.round(self.pos_green[:2],4)}")
        # print(f"[SPAWN DEBUG] magenta_start={np.round(self.magenta_start_pos[:2],4)}  cyan_start={np.round(self.cyan_start_pos[:2],4)}")
        # print(f"[SPAWN DEBUG] cyan_target={np.round(self.cyan_target_pos[:2],4)}  magenta_final={np.round(self.magenta_final_pos[:2],4)}")
        # print(f"[SPAWN DEBUG] start_magenta (with noise)={np.round(start_magenta[:2],4)}  start_cyan (with noise)={np.round(start_cyan[:2],4)}")

        # Add noise and Z-height
        start_magenta[:2] += np.random.uniform(-0.01, 0.01, 2)
        start_cyan[:2] += np.random.uniform(-0.01, 0.01, 2)
        start_magenta[2] = table_z + box_half_height
        start_cyan[2] = table_z + box_half_height

        default_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(self.box_magenta.joints[0], np.concatenate([start_magenta, default_quat]))
        self.sim.data.set_joint_qpos(self.box_cyan.joints[0], np.concatenate([start_cyan, default_quat]))

        # 3. Reset the State Machine
        self.task_phase = 0
        self._phase_hold = 0          # consecutive steps the current condition has been met
        self._phase_hold_required = 5 # steps a condition must hold before advancing

        self.sim.forward()

    def _get_phase_conditions(self):
        """Compute all position/gripper conditions without mutating any state."""
        magenta_pos = self.sim.data.body_xpos[self.box_magenta_body_id]
        cyan_pos    = self.sim.data.body_xpos[self.box_cyan_body_id]
        red_pos     = self.sim.data.body_xpos[self.cylinder_red_body_id]

        dist_threshold = 0.05
        magenta_on_red    = np.linalg.norm(magenta_pos[:2] - red_pos[:2]) < dist_threshold
        cyan_on_target    = np.linalg.norm(cyan_pos[:2]    - self.cyan_target_pos[:2]) < dist_threshold
        magenta_on_target = np.linalg.norm(magenta_pos[:2] - self.magenta_final_pos[:2]) < dist_threshold

        arm = self.robots[0].arms[0]
        gripper_qpos = self.sim.data.qpos[7]
        open_qpos    = self.robots[0].gripper[arm].init_qpos[0]
        gripper_is_open = np.abs(gripper_qpos) - np.abs(open_qpos) > 0.01

        return magenta_on_red, cyan_on_target, magenta_on_target, gripper_is_open

    def _setup_observables(self):
        observables = super()._setup_observables()
        pf = self.robots[0].robot_model.naming_prefix

        # 1. The main "object" array containing both cube positions
        @sensor(modality="object")
        def object_state(obs_cache):
            magenta_pos = np.array(self.sim.data.body_xpos[self.box_magenta_body_id])
            cyan_pos = np.array(self.sim.data.body_xpos[self.box_cyan_body_id])
            return np.concatenate([magenta_pos, cyan_pos])

        # 2. Gripper distance to the Magenta cube
        @sensor(modality="object")
        def gripper_to_magenta_pos(obs_cache):
            return obs_cache["object"][:3] - obs_cache[f"{pf}eef_pos"] if \
                "object" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

        # 3. Gripper distance to the Cyan cube
        @sensor(modality="object")
        def gripper_to_cyan_pos(obs_cache):
            return obs_cache["object"][3:6] - obs_cache[f"{pf}eef_pos"] if \
                "object" in obs_cache and f"{pf}eef_pos" in obs_cache else np.zeros(3)

        # Package them up
        sensors = [object_state, gripper_to_magenta_pos, gripper_to_cyan_pos]
        names = ["object", "gripper_to_magenta_pos", "gripper_to_cyan_pos"]

        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def _post_action(self, action):
        """Advance the phase state machine exactly once per step."""
        ret = super()._post_action(action)

        magenta_on_red, cyan_on_target, magenta_on_target, gripper_is_open = self._get_phase_conditions()

        # Map each phase to its advancement condition
        conditions = {
            0: magenta_on_red and gripper_is_open,
            1: cyan_on_target and gripper_is_open,
            2: magenta_on_target and gripper_is_open,
        }
        phase_labels = {
            0: "Phase 1 Complete: Magenta moved to empty Red slot.",
            1: "Phase 2 Complete: Cyan moved to Magenta's starting slot.",
            2: "Phase 3 Complete: Magenta moved to Cyan's starting slot.",
        }

        if self.task_phase in conditions:
            if conditions[self.task_phase]:
                self._phase_hold += 1
            else:
                self._phase_hold = 0  # reset hold counter if condition breaks

            if self._phase_hold >= self._phase_hold_required:
                print(phase_labels[self.task_phase])
                # Detailed debug dump on every phase transition
                magenta_pos = self.sim.data.body_xpos[self.box_magenta_body_id]
                cyan_pos    = self.sim.data.body_xpos[self.box_cyan_body_id]
                red_pos     = self.sim.data.body_xpos[self.cylinder_red_body_id]
                arm = self.robots[0].arms[0]
                gripper_qpos = self.sim.data.qpos[7]
                open_qpos    = self.robots[0].gripper[arm].init_qpos[0]
                self.task_phase += 1
                self._phase_hold = 0

        return ret

    def _check_success(self):
        """Read-only success check — does NOT mutate task_phase."""
        _, _, _, gripper_is_open = self._get_phase_conditions()
        return self.task_phase == 3 and gripper_is_open