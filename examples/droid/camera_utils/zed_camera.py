# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from copy import deepcopy

import cv2
import numpy as np
import os
import sys
import time

from camera_utils.parameters import hand_camera_id

def time_ms():
    return time.time_ns() // 1_000_000


def gather_zed_cameras():
    """Gather two OpenCV video devices and expose them as one stereo ZED camera."""
    # Try to identify two device indices via env, by-id, or probing
    # 1) Env vars override
    left_idx_env = os.getenv("ZED_LEFT_INDEX")
    right_idx_env = os.getenv("ZED_RIGHT_INDEX")
    indices: list[int] = []
    if left_idx_env is not None and right_idx_env is not None:
        try:
            indices = [int(left_idx_env), int(right_idx_env)]
        except Exception:
            indices = []

    # 2) Try v4l by-id entries containing 'ZED'
    if len(indices) < 2:
        try:
            by_id_dir = "/dev/v4l/by-id"
            if os.path.isdir(by_id_dir):
                candidates = []
                for name in sorted(os.listdir(by_id_dir)):
                    if ("ZED" in name) or ("zed" in name):
                        target = os.path.realpath(os.path.join(by_id_dir, name))
                        if target.startswith("/dev/video"):
                            try:
                                candidates.append(int(target.replace("/dev/video", "")))
                            except Exception:
                                pass
                candidates = sorted(set(candidates))
                if len(candidates) >= 2:
                    indices = candidates[:2]
        except Exception:
            pass

    # 3) Generic probe 0..7
    if len(indices) < 2:
        found = []
        for i in range(8):
            try:
                backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
                cap = cv2.VideoCapture(i, backend) if backend else cv2.VideoCapture(i)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        found.append(i)
                cap.release()
            except Exception:
                pass
        if len(found) >= 2:
            indices = found[:2]

    if len(indices) < 2:
        print("WARNING: Could not find two ZED video devices via OpenCV; skipping ZED.")
        return []

    return [ZedCamera(indices[0], indices[1], name="zed_opencv")]


resize_func_map = {"cv2": cv2.resize, None: None}


class ZedCamera:
    """
    Minimal OpenCV-based stereo camera wrapper to emulate ZED behavior.

    Treats two V4L2 devices as left/right and exposes the same public methods used elsewhere.
    Depth/pointcloud/recording are not implemented.
    """
    preferred_recording_extension = ".mp4"

    def __init__(self, left_index: int, right_index: int, name: str = "zed"):
        self.serial_number = str(name)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = "disabled"
        self._current_params = None

        self.left_index = int(left_index)
        self.right_index = int(right_index)
        self._left_cap: cv2.VideoCapture | None = None
        self._right_cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._record_mode: str | None = None  # "sbs" or "left"

        # Read-time configuration
        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.depth = False
        self.pointcloud = False
        self.resize_func = None
        self.skip_reading = False
        self.zed_resolution = (0, 0)
        self.resizer_resolution = (0, 0)
        self.latency = int(2.5 * (1e3 / 30))  # rough estimate @30 FPS

        print("Opening OpenCV ZED with indices:", self.left_index, self.right_index)

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image: bool = True,
        depth: bool = False,
        pointcloud: bool = False,
        concatenate_images: bool = False,
        resolution: tuple[int, int] = (0, 0),
        resize_func: str | None = None,
    ):
        # Non-permanent values
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permanent values
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    def set_calibration_mode(self):
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.zed_resolution = (0, 0)
        self.resizer_resolution = (0, 0)
        self._configure_opencv()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.zed_resolution = self.traj_resolution
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = (0, 0)
            self.resizer_resolution = self.traj_resolution

        self._configure_opencv()
        self.current_mode = "trajectory"

    def _configure_opencv(self):
        # Close existing caps
        self.disable_camera()

        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
        self._left_cap = cv2.VideoCapture(self.left_index, backend) if backend else cv2.VideoCapture(self.left_index)
        self._right_cap = cv2.VideoCapture(self.right_index, backend) if backend else cv2.VideoCapture(self.right_index)

        for cap in [self._left_cap, self._right_cap]:
            if not (cap and cap.isOpened()):
                continue
            # Prefer MJPG for higher throughput
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            # Apply requested capture resolution if provided
            w, h = self.zed_resolution
            if w and h:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
                except Exception:
                    pass

    def _maybe_resize(self, frame_np: np.ndarray):
        if self.resizer_resolution == (0, 0) or self.resize_func is None:
            return frame_np
        return self.resize_func(frame_np, self.resizer_resolution)

    def _maybe_write(self, left_img: np.ndarray | None, right_img: np.ndarray | None):
        if self._writer is None:
            return
        try:
            if self._record_mode == "sbs" and left_img is not None and right_img is not None:
                frame = np.concatenate([left_img, right_img], axis=1)
                self._writer.write(frame)
            elif left_img is not None:
                self._writer.write(left_img)
        except Exception:
            # ignore write errors to avoid crashing streaming
            pass

    def read_camera(self):
        if self.skip_reading or self._left_cap is None or self._right_cap is None:
            return {}, {}

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}

        # Read both cameras; tolerate one failing
        left_img = None
        right_img = None
        try:
            ok_l, frame_l = self._left_cap.read()
            if ok_l and frame_l is not None:
                left_img = frame_l
        except Exception:
            pass
        try:
            ok_r, frame_r = self._right_cap.read()
            if ok_r and frame_r is not None:
                right_img = frame_r
        except Exception:
            pass

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()
        now_ms = time_ms()
        timestamp_dict[self.serial_number + "_frame_received"] = now_ms
        timestamp_dict[self.serial_number + "_estimated_capture"] = now_ms - self.latency

        data_dict: dict[str, dict] = {}
        if self.image:
            if self.concatenate_images and (left_img is not None or right_img is not None):
                if left_img is None and right_img is not None:
                    sbs = np.concatenate([right_img, right_img], axis=1)
                elif right_img is None and left_img is not None:
                    sbs = np.concatenate([left_img, left_img], axis=1)
                elif left_img is not None and right_img is not None:
                    sbs = np.concatenate([left_img, right_img], axis=1)
                else:
                    sbs = None
                if sbs is not None:
                    sbs = self._maybe_resize(sbs)
                    data_dict["image"] = {self.serial_number: sbs}
            else:
                img_dict = {}
                if left_img is not None:
                    img_dict[self.serial_number + "_left"] = self._maybe_resize(left_img)
                if right_img is not None:
                    img_dict[self.serial_number + "_right"] = self._maybe_resize(right_img)
                if img_dict:
                    data_dict["image"] = img_dict

        # Write to file if recording
        self._maybe_write(left_img, right_img)

        return data_dict, timestamp_dict

    def disable_camera(self):
        # Stop recording first
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._record_mode = None
        if self._left_cap is not None:
            try:
                self._left_cap.release()
            except Exception:
                pass
        if self._right_cap is not None:
            try:
                self._right_cap.release()
            except Exception:
                pass
        self._left_cap = None
        self._right_cap = None
        self._current_params = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

    # Optional helpers for compatibility: no-op intrinsics and simple recording
    def get_intrinsics(self):
        return {}

    def start_recording(self, filename: str):
        # Determine a frame size by grabbing a single frame from opened caps
        if self._left_cap is None:
            self._configure_opencv()
        left_img = None
        right_img = None
        try:
            if self._left_cap is not None and self._left_cap.isOpened():
                ok, f = self._left_cap.read()
                if ok and f is not None:
                    left_img = f
            if self._right_cap is not None and self._right_cap.isOpened():
                ok, f = self._right_cap.read()
                if ok and f is not None:
                    right_img = f
        except Exception:
            pass

        # Choose recording mode
        self._record_mode = "sbs" if self.traj_concatenate_images and left_img is not None and right_img is not None else "left"

        if self._record_mode == "sbs" and left_img is not None and right_img is not None:
            h = max(left_img.shape[0], right_img.shape[0])
            w = left_img.shape[1] + right_img.shape[1]
        elif left_img is not None:
            h, w = left_img.shape[0], left_img.shape[1]
        else:
            # Fallback to a safe default
            h, w = 720, 1280

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        try:
            self._writer = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
        except Exception:
            self._writer = None

    def stop_recording(self):
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
        self._writer = None
        self._record_mode = None
