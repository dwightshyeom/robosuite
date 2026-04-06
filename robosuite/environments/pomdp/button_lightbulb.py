"""
ButtonLightbulb: A partial-observability benchmark task for MemoryDP.

N buttons and N lightbulbs are placed on a table. A random bijective mapping
from buttons to bulbs is generated each episode (and hidden from the robot).
Pressing button i TOGGLES the bulb secretly connected to it.

The goal: make exactly the target bulb lit, all other bulbs off.

To succeed, the robot must:
  1. Press buttons one by one, observing which bulb changes state each time.
  2. Remember which button connects to which bulb (hidden mapping).
  3. Once the target button is found, turn off any non-target bulbs
     that were accidentally lit during exploration by pressing those buttons again (toggle off).

Memory challenge: the robot should remember which buttons it already pressed
and what each one connects to. Re-pressing a known wrong button wastes a step
and signals a failure of working memory.

Arena layout (n=4, viewed from above, robot sits at +x side):

    +--------------------------------------------------------+
    | [btn_0]  [btn_1]  [btn_2]  [btn_3]   ← far side (-x)  |
    |                                                        |
    | [bulb_0] [bulb_1] [bulb_2] [bulb_3]  ← near robot (+x)|
    +--------------------------------------------------------+

    Robot reaches forward (-x direction) to press buttons.
    Bulbs are on the near side so the human operator can see them.

Partial observability: the button→bulb mapping is randomized and hidden.
Observable signals: which bulbs are currently lit, target bulb index.
"""
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_body,
    new_geom,
    new_joint,
    new_site,
)
from robosuite.utils.observables import Observable, sensor


class ButtonLightbulb(ManipulationEnv):
    """
    Press N buttons to toggle N lightbulbs via a hidden random mapping.

    Each episode a random permutation assigns each button to exactly one
    bulb. The robot must discover the correct button for the target bulb
    through trial-and-error, then restore all other bulbs to off.

    Success condition (default, require_exclusive=True):
        target bulb ON  AND  all other bulbs OFF.

    If require_exclusive=False:
        only the target bulb needs to be ON (easier variant).

    Args:
        robots (str or list of str): Robot specification.
        n_buttons (int): Number of buttons (= number of bulbs).
        button_spacing (float): Center-to-center spacing between buttons (m).
        button_x_offset (float): X offset of button row from table center (m).
        bulb_x_offset (float): X offset of bulb row from table center (m).
        button_press_threshold (float): Joint depression depth (m) to count as pressed.
        require_exclusive (bool): If True, all non-target bulbs must be OFF
            for success. If False, only target bulb needs to be ON.
        table_full_size (3-tuple): Full table dimensions (L, W, H).
        table_friction (3-tuple): Table friction coefficients.
        **kwargs: Additional ManipulationEnv arguments.
    """

    # Colors for bulbs
    _BULB_UNLIT_RGBA    = np.array([0.20, 0.20, 0.05, 1.0])   # dim yellow
    _BULB_LIT_RGBA      = np.array([1.00, 0.95, 0.10, 1.0])   # bright yellow
    _IND_TARGET_RGBA    = np.array([0.90, 0.15, 0.10, 1.0])   # red ring = target
    _IND_NONTARGET_RGBA = np.array([0.25, 0.25, 0.25, 1.0])   # gray ring

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        n_buttons=4,
        button_spacing=0.14,
        button_x_offset=-0.15,
        bulb_x_offset=0.15,
        button_press_threshold=0.005,
        require_exclusive=True,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
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
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        reward_shaping=False,
    ):
        assert n_buttons >= 2, "n_buttons must be at least 2"

        if controller_configs is None:
            robot_name = robots if isinstance(robots, str) else robots[0]
            controller_configs = load_composite_controller_config(robot=robot_name)
        if isinstance(controller_configs, dict) and "body_parts" in controller_configs:
            for arm_cfg in controller_configs["body_parts"].get("arms", {}).values():
                arm_cfg["type"] = "OSC_POSE"
                arm_cfg["input_type"] = "delta"

        self.n_buttons = n_buttons
        self.button_spacing = button_spacing
        self.button_x_offset = button_x_offset
        self.bulb_x_offset = bulb_x_offset
        self.button_press_threshold = button_press_threshold
        self.require_exclusive = require_exclusive

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.use_object_obs = use_object_obs

        # Geometry constants
        self._housing_radius = 0.038
        self._housing_height = 0.010
        self._btn_radius     = 0.033
        self._btn_cap_height = 0.008
        self._btn_dome_r     = 0.035
        self._bulb_sphere_r  = 0.022
        self._bulb_stem_r    = 0.008
        self._bulb_stem_h    = 0.030
        self._ind_radius     = 0.034   # flat indicator disc under bulb

        # Internal state — reset in _reset_internal
        self.mapping = list(range(n_buttons))   # mapping[i] = bulb index
        self.target_bulb_idx = 0
        self.bulb_states = [False] * n_buttons  # True = lit
        self._button_prev_pressed = [False] * n_buttons
        self._button_armed = [True] * n_buttons  # must release (spring relax) before next trigger
        self._bulb_toggle_counts = [0] * n_buttons  # how many times each bulb was toggled
        self._button_release_threshold = -self.button_press_threshold * 0.3  # near rest position
        self._memory_failure_latched = False   # latched when a clear memory failure is detected
        self._is_success = False
        self._is_done = False

        # Cached positions (world coords), filled in _reset_internal
        self._button_positions = []   # list of np.array([x, y, z])
        self._bulb_positions = []     # list of np.array([x, y, z])

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
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
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        """Load arena, N spring-loaded buttons, N visual lightbulbs."""
        super()._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Compute row center positions (table surface z)
        table_surface_z = self.table_offset[2] + self.table_full_size[2] / 2.0
        total_width = (self.n_buttons - 1) * self.button_spacing
        y_start = -total_width / 2.0

        self._button_positions = []
        self._bulb_positions = []

        for i in range(self.n_buttons):
            y = self.table_offset[1] + y_start + i * self.button_spacing

            btn_center = np.array([
                self.table_offset[0] + self.button_x_offset, y, table_surface_z
            ])
            bulb_center = np.array([
                self.table_offset[0] + self.bulb_x_offset, y, table_surface_z
            ])
            self._button_positions.append(btn_center)
            self._bulb_positions.append(bulb_center)

            self._add_single_button(mujoco_arena, i, btn_center)
            self._add_single_bulb(mujoco_arena, i, bulb_center)

        # No movable MuJoCo objects needed (buttons and bulbs live in arena)
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )

    def _add_single_button(self, arena, idx, center):
        """Add one spring-loaded button at `center` (table surface level).

        Design to prevent穿模 (clipping):
          - Large-radius dome (collision, radius=0.040) — wide contact surface
            so the gripper lands cleanly and doesn't slip through.
          - Very stiff spring (stiffness=800 N/m) resists the gripper.
          - Tight joint range [-threshold-0.003, 0.001] limits over-depression.
          - Detection: joint position < -threshold (button pressed DOWN),
            using rising-edge so only one event fires per press.
        """
        hr = 0.040      # dome/cap lateral radius
        dome_rz = 0.016  # dome vertical semi-axis (flat ellipsoid, equator at cylinder top)
        hh = 0.012   # housing half-height
        cap_hh = 0.006   # cap cylinder half-height (thin base)

        # --- Static housing (no collision, visual ring only) ---
        housing = new_body(name=f"button_{idx}_housing", pos=array_to_string(center))
        housing.append(new_geom(
            name=f"button_{idx}_housing_geom", type="cylinder",
            size=array_to_string([hr + 0.004, hh]),
            pos=array_to_string([0, 0, hh]),
            rgba="0.25 0.25 0.25 1", group="1",
            conaffinity="0", contype="0",
        ))
        arena.worldbody.append(housing)

        # --- Movable cap: thin base cylinder + large dome sphere ---
        # cap body rest position: base bottom flush with housing top
        cap_rest_z = 2 * hh + cap_hh
        cap = new_body(
            name=f"button_{idx}_cap",
            pos=array_to_string(center + np.array([0, 0, cap_rest_z])),
        )
        cap.append(ET.Element("inertial", attrib={
            "pos": "0 0 0", "mass": "0.08",
            "diaginertia": "1e-5 1e-5 1e-5",
        }))
        # Stiff spring with tight range: prevents over-depression (穿模)
        cap.append(new_joint(
            name=f"button_{idx}_joint",
            type="slide", axis="0 0 1",
            stiffness="800", damping="30",
            limited="true",
            range=array_to_string([-(self.button_press_threshold + 0.003), 0.001]),
        ))
        # Thin base cylinder (has collision, anchors the cap visually)
        cap.append(new_geom(
            name=f"button_{idx}_base", type="cylinder",
            size=array_to_string([hr, cap_hh]),
            pos="0 0 0",
            rgba="0.18 0.50 0.88 1", group="1",
            density="600", friction="0.5 0.005 0.0001",
        ))
        # Flat dome — main contact surface for the gripper (has collision)
        # Ellipsoid: lateral radius=hr, vertical semi-axis=dome_rz (flat cap)
        # Center at z=cap_hh so equator (max diameter) aligns with cylinder top
        cap.append(new_geom(
            name=f"button_{idx}_dome", type="ellipsoid",
            size=array_to_string([hr, hr, dome_rz]),
            pos=array_to_string([0, 0, cap_hh]),
            rgba="0.25 0.60 0.95 1", group="1",
            density="600", friction="0.5 0.005 0.0001",
        ))
        # Site at dome apex for observations
        cap.append(new_site(
            name=f"button_{idx}_top_site",
            pos=array_to_string([0, 0, cap_hh + dome_rz]),
            size="0.002", rgba="0 0 0 0",
        ))
        arena.worldbody.append(cap)

    def _add_single_bulb(self, arena, idx, center):
        """Add one static lightbulb visual at `center` (table surface level)."""
        sr = self._bulb_sphere_r
        str_ = self._bulb_stem_r
        sth = self._bulb_stem_h
        ir = self._ind_radius

        # Static body — no joint
        bulb_body = new_body(
            name=f"bulb_{idx}_body",
            pos=array_to_string(center),
        )

        # Indicator disc at base (colored ring: red = target, gray = non-target)
        bulb_body.append(new_geom(
            name=f"bulb_{idx}_indicator", type="cylinder",
            size=array_to_string([ir, 0.001]),
            pos=array_to_string([0, 0, 0.001]),
            rgba=array_to_string([*self._IND_NONTARGET_RGBA]),
            group="1", conaffinity="0", contype="0",
        ))

        # Stem
        bulb_body.append(new_geom(
            name=f"bulb_{idx}_stem", type="cylinder",
            size=array_to_string([str_, sth]),
            pos=array_to_string([0, 0, sth]),
            rgba="0.30 0.30 0.30 1", group="1",
            conaffinity="0", contype="0",
        ))

        # Bulb sphere (changes color when lit)
        bulb_body.append(new_geom(
            name=f"bulb_{idx}_sphere", type="sphere",
            size=array_to_string([sr]),
            pos=array_to_string([0, 0, 2 * sth + sr]),
            rgba=array_to_string([*self._BULB_UNLIT_RGBA]),
            group="1", conaffinity="0", contype="0",
        ))

        # Site for observations
        bulb_body.append(new_site(
            name=f"bulb_{idx}_site",
            pos=array_to_string([0, 0, 2 * sth + 2 * sr]),
            size="0.002", rgba="0 0 0 0",
        ))

        arena.worldbody.append(bulb_body)

    # ------------------------------------------------------------------
    # References & observables
    # ------------------------------------------------------------------

    def _setup_references(self):
        """Cache joint IDs for buttons and geom IDs for bulb spheres/indicators."""
        super()._setup_references()

        self._button_joint_ids = [
            self.sim.model.joint_name2id(f"button_{i}_joint")
            for i in range(self.n_buttons)
        ]
        self._button_body_ids = [
            self.sim.model.body_name2id(f"button_{i}_cap")
            for i in range(self.n_buttons)
        ]
        self._eef_site_id = list(self.robots[0].eef_site_id.values())[0]
        self._bulb_sphere_geom_ids = [
            self.sim.model.geom_name2id(f"bulb_{i}_sphere")
            for i in range(self.n_buttons)
        ]
        self._bulb_indicator_geom_ids = [
            self.sim.model.geom_name2id(f"bulb_{i}_indicator")
            for i in range(self.n_buttons)
        ]
        self._bulb_site_ids = [
            self.sim.model.site_name2id(f"bulb_{i}_site")
            for i in range(self.n_buttons)
        ]
        self._button_site_ids = [
            self.sim.model.site_name2id(f"button_{i}_top_site")
            for i in range(self.n_buttons)
        ]

    def _setup_observables(self):
        """
        Observable:
            - button_positions: all N button top-site positions (N×3 flattened).
            - bulb_positions: all N bulb site positions (N×3 flattened).
            - bulb_states: binary array (N,) — which bulbs are currently lit.
            - target_bulb_idx: index of the goal bulb (scalar).
            - gripper_to_button_{i}: per-button gripper distance.

        NOT observable:
            - mapping (button i → bulb j) — the hidden state each episode.
        """
        observables = super()._setup_observables()

        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def button_positions(obs_cache):
                return np.array([
                    self.sim.data.site_xpos[sid].copy()
                    for sid in self._button_site_ids
                ]).flatten()

            @sensor(modality=modality)
            def bulb_positions(obs_cache):
                return np.array([
                    self.sim.data.site_xpos[sid].copy()
                    for sid in self._bulb_site_ids
                ]).flatten()

            @sensor(modality=modality)
            def bulb_states(obs_cache):
                return np.array(self.bulb_states, dtype=np.float64)

            @sensor(modality=modality)
            def target_bulb_idx(obs_cache):
                return np.array([self.target_bulb_idx], dtype=np.float64)

            sensors = [button_positions, bulb_positions, bulb_states, target_bulb_idx]
            names = ["button_positions", "bulb_positions", "bulb_states", "target_bulb_idx"]

            # Per-button gripper-distance sensors
            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            for i in range(self.n_buttons):
                btn_key = f"button_{i}_pos"

                @sensor(modality=modality)
                def _btn_pos(obs_cache, _i=i):
                    return self.sim.data.site_xpos[self._button_site_ids[_i]].copy()

                _btn_pos.__name__ = btn_key
                sensors.append(_btn_pos)

                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes):
                    sensors.append(
                        self._get_obj_eef_sensor(
                            full_pf, btn_key,
                            f"{arm_pf}gripper_to_button_{i}",
                            modality,
                        )
                    )

            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self):
        """
        Randomize the button→bulb mapping and target bulb.
        Reset all bulb states to OFF, update visual colors.
        """
        super()._reset_internal()

        # Random permutation: mapping[i] = which bulb button i controls
        perm = list(self.rng.permutation(self.n_buttons))
        self.mapping = perm

        # Random target bulb
        self.target_bulb_idx = int(self.rng.integers(self.n_buttons))

        # Reset state
        self.bulb_states = [False] * self.n_buttons
        self._button_prev_pressed = [False] * self.n_buttons
        self._button_armed = [True] * self.n_buttons
        self._bulb_toggle_counts = [0] * self.n_buttons
        self._memory_failure_latched = False
        self._is_success = False
        self._is_done = False

        # Update visuals
        self._update_bulb_visuals()
        self._update_indicator_visuals()

    # ------------------------------------------------------------------
    # Action hooks
    # ------------------------------------------------------------------

    def _pre_action(self, action, policy_step=False):
        """Zero out rotation so gripper stays perpendicular to table."""
        action = np.array(action, dtype=np.float64)
        action[3:6] = 0.0
        return super()._pre_action(action, policy_step)

    def reward(self, action=None):
        return 0.0

    def _post_action(self, policy_step):
        ret = super()._post_action(policy_step)

        changed = False
        for i in range(self.n_buttons):
            joint_pos = self.sim.data.qpos[self._button_joint_ids[i]]
            currently_pressed = joint_pos < -self.button_press_threshold

            if not self._button_armed[i]:
                # Button was fired — wait for spring to relax before re-arming
                if joint_pos >= self._button_release_threshold:
                    self._button_armed[i] = True

            if self._button_armed[i] and currently_pressed and not self._button_prev_pressed[i]:
                # Armed + rising edge → fire
                bulb_idx = self.mapping[i]

                # (The old _memory_failure_latched logic was removed here to allow full exploration)

                self.bulb_states[bulb_idx] = not self.bulb_states[bulb_idx]
                self._bulb_toggle_counts[bulb_idx] += 1
                self._button_armed[i] = False  # disarm until spring relaxes
                changed = True

            self._button_prev_pressed[i] = currently_pressed

        if changed:
            self._update_bulb_visuals()

        return ret

    # ------------------------------------------------------------------
    # Visual updates
    # ------------------------------------------------------------------

    def _update_bulb_visuals(self):
        """Update sphere RGBA for all bulbs based on current bulb_states."""
        for j in range(self.n_buttons):
            rgba = self._BULB_LIT_RGBA if self.bulb_states[j] else self._BULB_UNLIT_RGBA
            self.sim.model.geom_rgba[self._bulb_sphere_geom_ids[j]] = rgba

    def _update_indicator_visuals(self):
        """Color indicator discs: red for target bulb, gray for others."""
        for j in range(self.n_buttons):
            rgba = (
                self._IND_TARGET_RGBA
                if j == self.target_bulb_idx
                else self._IND_NONTARGET_RGBA
            )
            self.sim.model.geom_rgba[self._bulb_indicator_geom_ids[j]] = rgba

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, vis_settings):
        """Visualize gripper proximity to nearest button."""
        super().visualize(vis_settings=vis_settings)

        if vis_settings["grippers"]:
            nearest = self._nearest_button_idx()
            if nearest is not None:
                self._visualize_gripper_to_target(
                    gripper=self.robots[0].gripper,
                    target=f"button_{nearest}_cap",
                    target_type="body",
                )

    # ------------------------------------------------------------------
    # Success / done
    # ------------------------------------------------------------------

    def _check_success(self):
        """
        Checks if the task was completed perfectly under EXHAUSTIVE EXPLORATION rules.
        Success ONLY happens if:
          1. ALL buttons have been pressed at least once.
          2. The target bulb is ON.
          3. All other bulbs are OFF.
          4. NO memory failures have occurred.
        """
        self.target_bulb_lit = self.bulb_states[self.target_bulb_idx]

        # NEW: Check if every single bulb has been toggled at least once
        self.all_explored = all(count >= 1 for count in self._bulb_toggle_counts)

        if self.require_exclusive:
            self.all_others_off = all(
                not state for j, state in enumerate(self.bulb_states)
                if j != self.target_bulb_idx
            )
        else:
            self.all_others_off = True

        # Ensure no failure has occurred before declaring success
        has_failed = self._check_failure()

        self._is_success = (not has_failed) and self.target_bulb_lit and self.all_others_off and self.all_explored
        return self._is_success

    def _check_failure(self):
        """
        Explicit failure conditions for the task (Memory Failure).
        Returns True if the user/policy makes an irreversible memory mistake.
        """
        self.memory_failure = False
        
        for j in range(self.n_buttons):
            if j == self.target_bulb_idx:
                # FAILURE 1: Target turned off after being lit (pressed goal button twice)
                if self._bulb_toggle_counts[j] >= 2:   
                    self.memory_failure = True
                    break
            else:
                # FAILURE 2: Non-target toggled too many times (on -> off -> on again)
                # (1=explored, 2=turned off correctly, 3=forgot and pressed again)
                if self._bulb_toggle_counts[j] >= 3:   
                    self.memory_failure = True
                    break
                    
        self._is_failure = self.memory_failure
        return self._is_failure

    def is_done(self):
        """
        Episode should terminate if the task succeeds OR fails.
        Robosuite automatically calls this inside env.step() to return the `done` flag.
        """
        success = self._check_success()
        failure = self._check_failure()
        
        self._is_done = success or failure
        return self._is_done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _nearest_button_idx(self):
        """Return the index of the button nearest to the gripper."""
        gripper_pos = self.sim.data.site_xpos[self._eef_site_id]
        dists = [
            np.linalg.norm(
                gripper_pos - self.sim.data.site_xpos[self._button_site_ids[i]]
            )
            for i in range(self.n_buttons)
        ]
        return int(np.argmin(dists))