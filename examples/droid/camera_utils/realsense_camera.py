# pyright: reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
from copy import deepcopy
from typing import TypedDict

import cv2
import numpy as np
import time
from camera_utils.parameters import hand_camera_id

import pyrealsense2 as rs

def time_ms():
    return time.time_ns() // 1_000_000


def gather_realsense_cameras():
    """Enumerate RealSense devices and return camera wrappers.

    Uses pyrealsense2 context to query connected devices.
    """
    all_cameras = []
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
    except Exception:
        return []

    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            continue
        all_cameras.append(RealsenseCamera(serial))
    return all_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

# Reasonable defaults for RealSense devices
class RSParams(TypedDict):
    color_resolution: tuple[int, int]
    depth_resolution: tuple[int, int]
    fps: int
    enable_depth: bool
    enable_ir: bool
    align_to_color: bool


standard_params: RSParams = {
    "color_resolution": (1280, 720),  # width, height
    "depth_resolution": (640, 480),
    "fps": 30,
    "enable_depth": False,
    "enable_ir": False,  # default conservative: color only
    "align_to_color": False,
}

advanced_params: RSParams = {
    "color_resolution": (1920, 1080),
    "depth_resolution": (1280, 720),
    "fps": 15,
    "enable_depth": False,
    "enable_ir": False,
    "align_to_color": False,
}


class RealsenseCamera:
    preferred_recording_extension = ".bag"
    def __init__(self, serial_number: str):
        # Save Parameters
        self.serial_number = str(serial_number)
        self.is_hand_camera = self.serial_number == hand_camera_id
        self.high_res_calibration = False
        self.current_mode = "disabled"
        self._current_params: RSParams | None = None

        # Runtime members
        self._pipeline: rs.pipeline | None = None
        self._profile: rs.pipeline_profile | None = None
        self._align: rs.align | None = None
        self._intrinsics: dict[str, dict] = {}
        self.latency: int = 0

        # Read-time configuration
        self.traj_image = True
        self.traj_concatenate_images = False
        self.traj_resolution = (0, 0)
        self.depth = False
        self.pointcloud = False
        self.resize_func = None
        self.skip_reading = False
        self.output_resolution = (0, 0)
        self.resizer_resolution = (0, 0)

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

    # Camera Modes
    def set_calibration_mode(self):
        # Parameters
        self.image = True
        self.concatenate_images = False
        self.skip_reading = False
        self.output_resolution = (0, 0)
        self.resizer_resolution = (0, 0)

        # Mode
        desired = advanced_params if self.high_res_calibration else standard_params
        if self._current_params != desired:
            self._configure_camera(desired)
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        # Parameters
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.output_resolution = self.traj_resolution
            self.resizer_resolution = (0, 0)
        else:
            self.output_resolution = (0, 0)
            self.resizer_resolution = self.traj_resolution

        # Mode
        if self._current_params != standard_params:
            self._configure_camera(standard_params)
        self.current_mode = "trajectory"

    def _configure_camera(self, init_params: RSParams):
        # Close Existing Camera
        self.disable_camera()

        color_w, color_h = init_params["color_resolution"]
        depth_w, depth_h = init_params["depth_resolution"]
        fps = int(init_params["fps"])

        # Build a sequence of increasingly conservative configs
        attempts: list[dict] = []
        # Attempt 1: requested params
        attempts.append({
            "color": (color_w, color_h, fps),
            "ir": "both" if init_params.get("enable_ir", False) else "none",
            "depth": bool(init_params.get("enable_depth", False)),
        })
        # Attempt 2: color + single IR
        attempts.append({"color": (1280, 720, fps), "ir": "left", "depth": False})
        # Attempt 3: color only 1280x720
        attempts.append({"color": (1280, 720, fps), "ir": "none", "depth": False})
        # Attempt 4: color only 640x480
        attempts.append({"color": (640, 480, fps), "ir": "none", "depth": False})
        # Attempt 5: color only 424x240
        attempts.append({"color": (424, 240, 30), "ir": "none", "depth": False})

        last_error: Exception | None = None
        for attempt in attempts:
            try:
                self._pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(self.serial_number)

                cw, ch, cfps = attempt["color"]
                cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, int(cfps))

                if attempt["ir"] in ("left", "both"):
                    cfg.enable_stream(rs.stream.infrared, 1, depth_w, depth_h, rs.format.y8, int(fps))
                if attempt["ir"] == "both":
                    cfg.enable_stream(rs.stream.infrared, 2, depth_w, depth_h, rs.format.y8, int(fps))
                if attempt["depth"]:
                    cfg.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, int(fps))

                assert self._pipeline is not None
                self._profile = self._pipeline.start(cfg)

                # Optional alignment
                self._align = rs.align(rs.stream.color) if init_params.get("align_to_color", False) else None

                # Save current params and latency estimate
                self._current_params = init_params
                self.latency = int(2.5 * (1e3 / max(1, int(cfps))))

                # Save intrinsics for available video streams
                self._intrinsics = {}
                try:
                    assert self._profile is not None
                    # Try IR streams first
                    try:
                        ir1 = self._profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
                        self._intrinsics[self.serial_number + "_left"] = self._process_intrinsics_ir(ir1)
                    except Exception:
                        pass
                    try:
                        ir2 = self._profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile().get_intrinsics()
                        self._intrinsics[self.serial_number + "_right"] = self._process_intrinsics_ir(ir2)
                    except Exception:
                        pass
                    # Color as left if IR not present
                    try:
                        color = self._profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                        if self.serial_number + "_left" not in self._intrinsics:
                            self._intrinsics[self.serial_number + "_left"] = self._process_intrinsics_ir(color)
                    except Exception:
                        pass
                except Exception:
                    self._intrinsics = {}

                # Success
                break
            except Exception as e:  # try next attempt
                last_error = e
                try:
                    if self._pipeline is not None:
                        self._pipeline.stop()
                except Exception:
                    pass
                self._pipeline = None
                self._profile = None
                self._align = None
        else:
            # If no attempt succeeded, raise the last error
            if last_error is not None:
                raise last_error
            raise RuntimeError("Failed to start RealSense pipeline with any configuration")

    # Calibration Utilities
    def _process_intrinsics_ir(self, intr):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        # rs intrinsics model has 5 coefficients (k1,k2,p1,p2,k3) or more depending on model
        intrinsics["distCoeffs"] = np.array(list(intr.coeffs))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    # Recording Utilities
    def start_recording(self, filename: str):
        # RealSense requires enabling recording on config before starting the pipeline.
        # We restart the pipeline with recording enabled to the requested file path.
        current_params: RSParams = self._current_params if self._current_params is not None else standard_params
        self.disable_camera()
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial_number)
        cfg.enable_record_to_file(filename)

        color_w, color_h = current_params["color_resolution"]
        depth_w, depth_h = current_params["depth_resolution"]
        fps = int(current_params["fps"])
        cfg.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        if current_params.get("enable_ir", False):
            cfg.enable_stream(rs.stream.infrared, 1, depth_w, depth_h, rs.format.y8, fps)
            cfg.enable_stream(rs.stream.infrared, 2, depth_w, depth_h, rs.format.y8, fps)
        if current_params.get("enable_depth", False):
            cfg.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
        assert self._pipeline is not None
        self._profile = self._pipeline.start(cfg)

    def stop_recording(self):
        # Simply restart without recording
        params = self._current_params or standard_params
        self._configure_camera(params)

    # Basic Camera Utilities
    def _maybe_resize(self, frame_np: np.ndarray):
        if self.resizer_resolution == (0, 0) or self.resize_func is None:
            return frame_np
        return self.resize_func(frame_np, self.resizer_resolution)

    def read_camera(self):
        # Skip if read unnecessary
        if self.skip_reading or self._pipeline is None:
            return {}, {}

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}

        try:
            assert self._pipeline is not None
            frames = self._pipeline.wait_for_frames()
        except Exception:
            # Camera read failed; return empty
            timestamp_dict[self.serial_number + "_read_end"] = time_ms()
            return {}, timestamp_dict

        if self._align is not None:
            frames = self._align.process(frames)

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        received_time = None
        try:
            # Use color frame timestamp if available
            cf = frames.get_color_frame()
            if cf:
                received_time = cf.get_timestamp()
            else:
                ir = frames.get_infrared_frame(1)
                if ir:
                    received_time = ir.get_timestamp()
        except Exception:
            pass
        if received_time is not None:
            # pyrealsense2 timestamp in ms
            timestamp_dict[self.serial_number + "_frame_received"] = received_time
            timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        data_dict: dict[str, dict] = {}

        if self.image:
            # Collect left/right images if available; fall back to color as left
            left_img = None
            right_img = None

            try:
                ir_left = frames.get_infrared_frame(1)
                if ir_left:
                    ir_left_np = np.asanyarray(ir_left.get_data())
                    # convert to BGR 3-channel for consistency
                    left_img = cv2.cvtColor(ir_left_np, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass
            try:
                ir_right = frames.get_infrared_frame(2)
                if ir_right:
                    ir_right_np = np.asanyarray(ir_right.get_data())
                    right_img = cv2.cvtColor(ir_right_np, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

            if left_img is None:
                try:
                    color = frames.get_color_frame()
                    if color:
                        left_img = np.asanyarray(color.get_data())  # already BGR8
                except Exception:
                    pass

            if self.concatenate_images and left_img is not None:
                if right_img is None:
                    right_img = left_img
                sbs = np.concatenate([left_img, right_img], axis=1)
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

        if self.depth:
            try:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_np = np.asanyarray(depth_frame.get_data())
                    if self.resizer_resolution != (0, 0) and self.resize_func is not None:
                        depth_np = self.resize_func(depth_np, self.resizer_resolution)
                    data_dict.setdefault("depth", {})[self.serial_number] = depth_np
            except Exception:
                pass

        # Pointcloud not implemented in this wrapper

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._align = None
        self._current_params = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
