from pynput.keyboard import Key, Listener
import numpy as np
from robosuite.devices import Device
import robosuite.utils.transform_utils as T

class PhantomOmni_ORI(Device):
    """
    A custom device class for the Phantom Omni.
    """
    def __init__(self, env, ros_node, pos_sensitivity=1.0, rot_sensitivity=1.0):
        super().__init__(env)
        self.ros_node = ros_node
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity 
        
        self._reset_internal_state()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.prev_largest_roll = 0.0
        self.prev_largest_pitch = 0.0
        self.prev_largest_yaw = 0.0

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.last_pos = None
        self.last_master_R = None  # <-- Changed from Euler to Matrix
        self._grasp = False
        self._reset_state = 0
        self._enabled = False
        self._clutch_active = False

    def start_control(self):
        self._reset_internal_state()
        self._enabled = True

    def get_controller_state(self):
        if self.ros_node.master_position is None or not self._enabled:
            return dict(dpos=np.zeros(3), rotation=np.eye(3), raw_drotation=np.zeros(3), grasp=False, reset=self._reset_state, base_mode=0)

        current_pos = self.ros_node.master_position.copy()

        # 1. READ ORIENTATION FROM ROS NODE AS QUATERNION
        if hasattr(self.ros_node, 'master_quat') and self.ros_node.master_quat is not None:
            current_quat = self.ros_node.master_quat.copy()
            # Convert directly to Rotation Matrix
            master_R = T.quat2mat(current_quat)
            # print(f"Current Quaternion: {current_quat}, Converted Rotation Matrix:\n{master_R}")
        else:
            master_R = np.eye(3)

        # 2. INITIALIZATION
        if self.last_pos is None:
            mujoco_x = 2*current_pos[2]  
            mujoco_y = 2*current_pos[0]  
            mujoco_z = 1*current_pos[1]  
            self.last_pos = np.array([mujoco_x, mujoco_y, mujoco_z])
            self.last_master_R = master_R  # <-- Save initial Matrix

        # 3. POSITION MAPPING
        mujoco_x = 2*current_pos[2]  
        mujoco_y = 2*current_pos[0]  
        mujoco_z = 1*current_pos[1]  
        mapped_pos = np.array([mujoco_x, mujoco_y, mujoco_z])
        dpos = mapped_pos - self.last_pos
        self.last_pos = mapped_pos

        # =======================================================
        # 4. ROTATION MAPPING (BODY-FRAME MATH)
        # =======================================================
        
        # A. Calculate Delta in World Frame: $\Delta R_{world} = R_{current} \times R_{last}^T$
        delta_R_world = master_R @ self.last_master_R.T
        
        # B. Project into Local Body Frame: $\Delta R_{local} = R_{last}^T \times \Delta R_{world} \times R_{last}$
        delta_R_local = self.last_master_R.T @ delta_R_world @ self.last_master_R
        # print(f"\rDelta Rotation in World Frame:\n{delta_R_world}\nDelta Rotation in Local Frame:\n{delta_R_local}")
        # C. Extract Local Euler Angles (Delta Roll, Delta Pitch, Delta Yaw)
        drotation_local = np.array(T.mat2euler(delta_R_local))
        # print(f"Delta Rotation (Local Euler Angles): {drotation_local}")
        
        # D. Map the axes to match MuJoCo
        # Swap these indices (0, 1, 2) just like you calibrated the position!
        mapped_drotation = np.array([
             1.0 * drotation_local[1],  # Mapped Roll  <- Omni Pitch
             1.0 * drotation_local[2],   # Mapped Yaw   <- Omni Yaw
            -1.0 * drotation_local[0],  # Mapped Pitch <- Negative Omni Roll
        ])
        largest_roll = max(np.abs(mapped_drotation[0]), self.prev_largest_roll)
        self.prev_largest_roll = largest_roll
        largest_pitch = max(np.abs(mapped_drotation[1]), self.prev_largest_pitch)
        self.prev_largest_pitch = largest_pitch
        largest_yaw = max(np.abs(mapped_drotation[2]), self.prev_largest_yaw)
        self.prev_largest_yaw = largest_yaw
        print(f"Mapped Delta Rotation Roll: {mapped_drotation[0]:6.3f}, Pitch: {mapped_drotation[1]:6.3f}, Yaw: {mapped_drotation[2]:6.3f}, largest_roll: {largest_roll:6.3f}, largest_pitch: {largest_pitch:6.3f}, largest_yaw: {largest_yaw:6.3f}", end="\r")
        # Update baseline for the next frame
        self.last_master_R = master_R

        # =======================================================
        # CLUTCH
        # =======================================================
        if not self._clutch_active:
            dpos = np.zeros(3)
            mapped_drotation = np.zeros(3)  

        return dict(
            dpos=dpos * self.pos_sensitivity,
            rotation=master_R,  
            raw_drotation=mapped_drotation * self.rot_sensitivity, 
            grasp=self._grasp,                    
            reset=self._reset_state,
            base_mode=0
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        dpos = dpos * 125
        dpos = np.clip(dpos, -1, 1)

        # Scale / clamp rotation so orientation control is responsive and stable
        drotation = drotation * 10
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    def on_press(self, key):
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = True
        elif key == Key.esc:
            self._reset_state = 1
            self._enabled = False

    def on_release(self, key):
        if key == Key.shift or key == Key.shift_l or key == Key.shift_r:
            self._clutch_active = False
        elif key == Key.space:
            self._grasp = not self._grasp