from copy import deepcopy
from math import trunc
from this import s
import time
from camera_utils.multi_camera_wrapper import MultiCameraWrapper
from camera_utils.info import camera_type_dict

import gymnasium as gym
import numpy as np
import robotic as ry


class RaiEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", gripper_action_space=None, camera_kwargs={}, do_reset=True):
        # Initialize Gym Environment
        super().__init__()

        self.C = ry.Config()
        self.C.addFile("$RAI_PATH/scenarios/pandaSingle.g")
        self.q0 = self.C.getJointState()
        # Define Action Space #
        # assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        # self.action_space = "joint_velocity"  # pyright: ignore[reportAttributeAccessIssue]
        self.gripper_action_space = "velocity"
        self.check_action_range = "velocity" in action_space

        self.prev_joint_torques_computed = list(np.zeros(7))
        self.prev_joint_torques_computed_safened = list(np.zeros(7))

        # Robot Configuration
        self.reset_joints = self.q0
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7
        self.control_hz = 15

        self.bot = ry.BotOp(self.C, useRealRobot=False)

        self.camera_reader = MultiCameraWrapper(camera_kwargs)
        self.camera_type_dict = camera_type_dict

        # Reset Robot
        if do_reset:
            self.reset()

    # def multi_dynamics(self, q, q_dot: np.ndarray):
    #     q_new = [q]
    #     dt = 1/self.control_hz
    #     for i in range(len(q_dot)):
    #         # create position path
    #         q_new.append(self.dynamics(q_new[-1], q_dot[i])[0])
    #     q_new = np.array(q_new)[1:] # remove initial position
    #     T = [dt*len(q_dot)] # time to reach each joint position
    #     return q_new, T

    def dynamics(self, q, q_dot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dt = 1/self.control_hz
        q_new = deepcopy(q)
        q_new[:-1] = q[:-1] + q_dot[:-1] * dt
        q_new[-1] = q_dot[-1]   # gripper pos
        delta_q = q_new - q
        return q_new, delta_q

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF +1 # 

        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        q = np.append(self.bot.get_q(), np.expand_dims(self.bot.getGripperPos(ry._left), axis=-1), axis=-1)
        q_new, delta_q = self.dynamics(q, action)
        q_new_joints = q_new[:-1]
        q_new_gripper = q_new[-1]
        self.bot.move([q_new_joints], times=[1/self.control_hz])
        # self.bot.wait(self.C, forTimeToEnd=False)

        # Return Action Info
        obs, reward, terminated, truncated, info = self.get_observation()
        return obs, reward, terminated, truncated, info

    def reset(self, randomize=False, seed=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        del self.bot
        self.bot = ry.BotOp(self.C, useRealRobot=False)
        
    def read_cameras(self):
        camera_obs, camera_timestamp = self.camera_reader.read_cameras()
        return camera_obs, camera_timestamp

    def get_state(self):
        read_start = 0
        timestamp_dict = {}
        timestamp_dict["read_start"] = self.bot.get_t()
        state_dict = {
            "cartesian_position": self.C.getFrame("l_gripper").getPosition(),
            "gripper_position":  self.bot.getGripperPos(ry._left),
            "joint_positions": self.bot.get_q(),
            "joint_velocities":  self.bot.get_qDot(),
            "joint_torques_computed": self.bot.get_tauExternal(),
            "prev_joint_torques_computed": self.prev_joint_torques_computed,
            "prev_joint_torques_computed_safened": self.prev_joint_torques_computed_safened,
            "prev_controller_latency_ms": 0,
            "prev_command_successful": True if self.bot.getTimeToEnd()<=0 else False,
        }
        timestamp_dict["read_end"] = self.bot.get_t()

        return state_dict, {}

    def get_camera_extrinsics(self, state_dict=None):
        # Adjust gripper camere by current pose
        # extrinsics = deepcopy(self.calibration_dict)
        self.bot.sync(self.C, .1)
        quat = self.C.eval(ry.FS.pose, ["cameraWrist"])[0]
        
        extrinsics = ry.Quaternion().set(quat[3:]).getMatrix()

        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        obs_dict["camera_type"] = deepcopy(self.camera_type_dict)
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict, None, None, None, None
