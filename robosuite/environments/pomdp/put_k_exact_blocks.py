"""
PutKExactBlocks: A partial-observability benchmark task for MemoryDP.

The robot must place exactly k blocks into an opaque box, then press a red
button to signal completion. To enforce true partial observability and reduce
state dimension, ONLY ONE BLOCK exists in the simulation. When placed in the 
box, it is immediately teleported back to the supply position.
"""
import xml.etree.ElementTree as ET
import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.mjcf_utils import CustomMaterial, array_to_string, new_body, new_geom, new_joint, new_site
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

class PutKExactBlocks(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        k=3,
        box_size=(0.07, 0.07, 0.06),
        box_pos_offset=(0.0, 0.20),
        supply_pos_offset=(0.0, 0.0),
        button_pos_offset=(0.0, -0.20),
        button_press_threshold=0.005,
        placement_dwell_steps=3,
        max_respawns=8,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=False,
        use_object_obs=True,
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
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
        seed=None,
        reward_shaping=False,
    ):
        if controller_configs is None:
            robot_name = robots if isinstance(robots, str) else robots[0]
            controller_configs = load_composite_controller_config(robot=robot_name)
        if isinstance(controller_configs, dict) and "body_parts" in controller_configs:
            for arm_cfg in controller_configs["body_parts"].get("arms", {}).values():
                arm_cfg["type"] = "OSC_POSE"
                arm_cfg["input_type"] = "delta"

        self.k = k
        self.box_size = np.array(box_size)
        self.box_pos_offset = np.array(box_pos_offset)
        self.supply_pos_offset = np.array(supply_pos_offset)
        self.button_pos_offset = np.array(button_pos_offset)
        self.button_press_threshold = button_press_threshold
        self.placement_dwell_steps = placement_dwell_steps
        self.max_respawns = max_respawns

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        # internal state
        self.blocks_in_box = 0
        self.button_pressed = False
        self._in_box_dwell = 0
        self._respawn_count = 0
        self._respawn_limit_hit = False

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

    def _load_model(self):
        super()._load_model()
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        self._add_opaque_box(mujoco_arena)
        self._add_button(mujoco_arena)

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        block_material = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # Only ONE block is created
        self.block = BoxObject(
            name="block_0",
            size_min=[0.018, 0.018, 0.018],
            size_max=[0.020, 0.020, 0.020],
            rgba=[0.2, 0.4, 0.9, 1],
            material=block_material,
        )

        supply_pos = self.table_offset + np.array([self.supply_pos_offset[0], self.supply_pos_offset[1], 0.0])
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.block)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="BlockSupplySampler",
                mujoco_objects=self.block,
                x_range=[-0.01, 0.01],
                y_range=[-0.01, 0.01],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=supply_pos,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.block],
        )

    # _add_opaque_box and _add_button remain EXACTLY the same as your original code
    def _add_opaque_box(self, arena):
        bx, by, bz = self.box_size
        wt = 0.002
        box_center = self.table_offset + np.array([self.box_pos_offset[0], self.box_pos_offset[1], bz + wt])
        box_body = new_body(name="opaque_box", pos=array_to_string(box_center))
        box_rgba = "0.25 0.25 0.25 1"
        box_body.append(new_geom(name="box_bottom", type="box", size=array_to_string([bx + wt, by + wt, wt]), pos=array_to_string([0, 0, -bz]), rgba=box_rgba, group="1", friction="1 0.005 0.0001"))
        box_body.append(new_geom(name="box_front", type="box", size=array_to_string([wt, by + wt, bz]), pos=array_to_string([bx + wt, 0, 0]), rgba=box_rgba, group="1", friction="1 0.005 0.0001"))
        box_body.append(new_geom(name="box_back", type="box", size=array_to_string([wt, by + wt, bz]), pos=array_to_string([-(bx + wt), 0, 0]), rgba=box_rgba, group="1", friction="1 0.005 0.0001"))
        box_body.append(new_geom(name="box_left", type="box", size=array_to_string([bx, wt, bz]), pos=array_to_string([0, by + wt, 0]), rgba=box_rgba, group="1", friction="1 0.005 0.0001"))
        box_body.append(new_geom(name="box_right", type="box", size=array_to_string([bx, wt, bz]), pos=array_to_string([0, -(by + wt), 0]), rgba=box_rgba, group="1", friction="1 0.005 0.0001"))
        box_body.append(new_site(name="box_top_center", pos=array_to_string([0, 0, bz]), size="0.002", rgba="0 0 0 0"))
        arena.worldbody.append(box_body)

    def _add_button(self, arena):
        housing_radius, housing_height = 0.032, 0.010
        btn_radius, btn_dome_radius, btn_cap_height = 0.028, 0.028, 0.006
        btn_center = self.table_offset + np.array([self.button_pos_offset[0], self.button_pos_offset[1], 0.0])
        housing_body = new_body(name="button_housing", pos=array_to_string(btn_center))
        housing_body.append(new_geom(name="button_housing_geom", type="cylinder", size=array_to_string([housing_radius, housing_height]), pos=array_to_string([0, 0, housing_height]), rgba="0.35 0.35 0.35 1", group="1", conaffinity="0", contype="0"))
        arena.worldbody.append(housing_body)
        cap_rest_z = 2 * housing_height + btn_cap_height
        cap_body = new_body(name="button_cap", pos=array_to_string(btn_center + np.array([0, 0, cap_rest_z])))
        inertial = ET.Element("inertial", attrib={"pos": "0 0 0", "mass": "0.05", "diaginertia": "1e-5 1e-5 1e-5"})
        cap_body.append(inertial)
        cap_body.append(new_joint(name="button_joint", type="slide", axis="0 0 1", stiffness="40", damping="3", limited="true", range=array_to_string([-(self.button_press_threshold + 0.005), 0.002])))
        cap_body.append(new_geom(name="button_cap_base", type="cylinder", size=array_to_string([btn_radius, btn_cap_height]), pos="0 0 0", rgba="0.85 0.12 0.12 1", group="1", density="500", friction="1 0.005 0.0001"))
        cap_body.append(new_geom(name="button_cap_dome", type="sphere", size=array_to_string([btn_dome_radius]), pos=array_to_string([0, 0, btn_cap_height * 0.3]), rgba="0.9 0.15 0.15 1", group="1", density="500", friction="1 0.005 0.0001"))
        cap_body.append(new_site(name="button_top_site", pos=array_to_string([0, 0, btn_cap_height + btn_dome_radius * 0.5]), size="0.002", rgba="0 0 0 0"))
        arena.worldbody.append(cap_body)

    def _box_center_pos(self):
        return np.array(self.sim.data.get_site_xpos("box_top_center"))

    def _is_button_pressed(self):
        return self.sim.data.get_joint_qpos("button_joint") < -self.button_press_threshold
    
    def _is_gripper_open(self):
        arm = self.robots[0].arms[0]
        gripper_qpos = self.sim.data.qpos[8]
        open_qpos = self.robots[0].gripper[arm].init_qpos[0] * 2
        gripper_is_open = np.abs(np.abs(open_qpos) - np.abs(gripper_qpos)) < 0.005

        return gripper_is_open

    def _button_pos(self):
        return np.array(self.sim.data.get_site_xpos("button_top_site"))

    def _is_block_out_of_reach(self):
        block_pos = self.sim.data.body_xpos[self.block_body_id]
        table_x, table_y, _ = self.table_full_size
        table_z = self.table_offset[2]
        if block_pos[2] < table_z - 0.05: return True
        if abs(block_pos[0] - self.table_offset[0]) > table_x / 2 + 0.05: return True
        if abs(block_pos[1] - self.table_offset[1]) > table_y / 2 + 0.05: return True
        return False

    def _respawn_block_to_supply(self):
        """Teleports the single block back to the supply coordinate."""
        supply_pos = self.table_offset + np.array([
            self.supply_pos_offset[0], self.supply_pos_offset[1], 0.01
        ])
        supply_quat = np.array([1, 0, 0, 0])
        self.sim.data.set_joint_qpos(
            self.block.joints[0],
            np.concatenate([supply_pos, supply_quat]),
        )
        self.sim.data.set_joint_qvel(self.block.joints[0], np.zeros(6))
        self.sim.forward()

    def _is_block_in_box(self):
        block_pos = self.sim.data.body_xpos[self.block_body_id]
        box_center = self._box_center_pos()
        bx, by, bz = self.box_size
        in_x = abs(block_pos[0] - box_center[0]) < bx
        in_y = abs(block_pos[1] - box_center[1]) < by
        # in_z = (box_center[2] - 2 * bz) < block_pos[2] < (box_center[2] + 0.02)
        in_z = (box_center[2] - 2 * bz) < block_pos[2] < (box_center[2])
        return in_x and in_y and in_z

    def _setup_references(self):
        super()._setup_references()
        self.block_body_id = self.sim.model.body_name2id(self.block.root_body)

    def _setup_observables(self):
        observables = super()._setup_observables()
        if self.use_object_obs:
            modality = "object"

            @sensor(modality=modality)
            def current_block_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.block_body_id])

            @sensor(modality=modality)
            def current_block_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.block_body_id]), to="xyzw")

            @sensor(modality=modality)
            def box_pos(obs_cache):
                return self._box_center_pos()

            @sensor(modality=modality)
            def button_pos(obs_cache):
                return self._button_pos()

            sensors = [current_block_pos, current_block_quat, box_pos, button_pos]

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])
            sensors += [
                self._get_obj_eef_sensor(full_pf, "current_block_pos", f"{arm_pf}gripper_to_block_pos", modality)
                for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
            ]

            names = [s.__name__ for s in sensors]
            for name, s in zip(names, sensors):
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self.blocks_in_box = 0
        self.button_pressed = False
        self._in_box_dwell = 0
        self._respawn_count = 0
        self._respawn_limit_hit = False

        if not self.deterministic_reset:
            self._respawn_block_to_supply()

    def _pre_action(self, action, policy_step=False):
        action = np.array(action, dtype=np.float64)
        action[3:6] = 0.0
        return super()._pre_action(action, policy_step)

    def reward(self, action=None):
        return 0.0

    def _post_action(self, policy_step):
        ret = super()._post_action(policy_step)
        self.gripper_is_open = self._is_gripper_open()

        # Check if block entered the box
        if self._is_block_in_box():
            self._in_box_dwell += 1
        else:
            self._in_box_dwell = 0

        # Successful placement logic
        if self._in_box_dwell >= self.placement_dwell_steps and self.gripper_is_open:
            self.blocks_in_box += 1
            self._in_box_dwell = 0
            self._respawn_count = 0 
            print(f"Total in box: {self.blocks_in_box}.")
            
            # Immediately teleport the SAME block back to the supply spot
            self._respawn_block_to_supply()

        # Respawn block if it fell off the table
        elif self._is_block_out_of_reach():
            self._respawn_count += 1
            if self._respawn_count > self.max_respawns:
                self._respawn_limit_hit = True
            self._in_box_dwell = 0
            self._respawn_block_to_supply()

        # Check button press
        if not self.button_pressed and self._is_button_pressed():
            self.button_pressed = True
            print(f"[PutKExactBlocks] BUTTON PRESSED! Placements: {self.blocks_in_box}, Target (k): {self.k}, "
                  f"Success: {self.blocks_in_box == self.k}")

        return ret

    def visualize(self, vis_settings):
        super().visualize(vis_settings=vis_settings)
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.block,
            )

    def _check_success(self):
        """
        Checks if the task was completed perfectly.
        Success ONLY happens if the button is pressed and exactly k blocks are in the box.
        """
        self.correct_count = (self.blocks_in_box == self.k)
        self._is_success = self.button_pressed and self.correct_count
        return self._is_success

    def _check_failure(self):
        """
        Explicit failure conditions for the task.
        Returns True if the user/policy makes an irreversible mistake.
        """
        # 1. Pressed the button when blocks_in_box is not exactly k
        premature_press = self.button_pressed and (self.blocks_in_box != self.k)
        
        # 2. Placed too many blocks in the box
        overflow = (self.blocks_in_box > self.k)
        
        # 3. Dropped the block off the table too many times
        respawn_limit = self._respawn_limit_hit
        
        self._is_failure = premature_press or overflow or respawn_limit
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