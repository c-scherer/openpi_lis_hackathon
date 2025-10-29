import argparse
import os
import sys
from typing import Optional

import cv2


# Ensure the 'droid' package is importable (used by the camera wrapper)
PROJECT_ROOT = "/home/hackathon/openpi_lis_hackathon"
if os.path.isdir(os.path.join(PROJECT_ROOT, "droid")):
    droid_pkg_root = os.path.join(PROJECT_ROOT, "droid")
    if droid_pkg_root not in sys.path:
        sys.path.append(droid_pkg_root)


# Import the RealSense camera wrapper from the examples module
from camera_utils.realsense_camera import gather_realsense_cameras  # noqa: E402


def save_first_available_image(save_path: str, frames: int = 1) -> int:
    cams = gather_realsense_cameras()
    if not cams:
        print("No RealSense cameras found.")
        return 1

    # Use the first detected camera
    cam = cams[0]
    print(f"Opening RealSense camera {cam.serial_number}")

    # Configure reading parameters
    cam.set_reading_parameters(
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),  # use native resolution
        resize_func="cv2",
    )
    cam.set_trajectory_mode()

    saved = 0
    for i in range(frames):
        data_dict, _ = cam.read_camera()
        if not data_dict:
            continue

        images = data_dict.get("image", {})
        if not images:
            continue

        # Prefer left image if available, else take any image
        key: Optional[str] = next((k for k in images.keys() if k.endswith("_left")), None)
        if key is None:
            key = next(iter(images.keys()))

        frame = images[key]
        out_path = save_path if frames == 1 else f"realsense_test_{i}.png"
        ok = cv2.imwrite(out_path, frame)
        if ok:
            print(f"Saved {frame.shape} to {out_path}")
            saved += 1
        else:
            print(f"Failed to save image to {out_path}")

    cam.disable_camera()
    if saved == 0:
        print("Did not capture any images.")
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description="Quick RealSense USB camera test")
    parser.add_argument("--save_path", type=str, default="realsense_test.png", help="Output image path")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to save (>=1)")
    args = parser.parse_args()

    code = save_first_available_image(args.save_path, max(1, int(args.frames)))
    raise SystemExit(code)


if __name__ == "__main__":
    main()


