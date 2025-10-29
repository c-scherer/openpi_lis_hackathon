#!/usr/bin/env python3
"""
Quick OpenCV camera probe for dual-camera setups.

- Probes a range of indices (default: 0..7) and reports which are usable.
- If at least two cameras are found, opens the first two simultaneously,
  reads frames for a few seconds, and saves one snapshot per camera next to this script.

Run:
  uv run examples/droid/test_opencv_cameras.py --max-indices 8 --width 1280 --height 720
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover
    print("ERROR: OpenCV (cv2) is not installed.")
    print("Install with: uv add opencv-python")
    raise


def list_v4l_symlinks() -> List[str]:
    by_id_dir = "/dev/v4l/by-id"
    if not os.path.isdir(by_id_dir):
        return []
    entries = []
    for name in sorted(os.listdir(by_id_dir)):
        path = os.path.join(by_id_dir, name)
        try:
            target = os.path.realpath(path)
            entries.append(f"{path} -> {target}")
        except Exception:
            entries.append(f"{path} -> <unreadable>")
    return entries


def try_open_once(index: int, width: int, height: int, prefer_v4l2: bool) -> Tuple[bool, Optional[Tuple[int, int]]]:
    backend_flag = cv2.CAP_V4L2 if prefer_v4l2 else 0
    cap = cv2.VideoCapture(index, backend_flag) if backend_flag else cv2.VideoCapture(index)
    if not cap.isOpened():
        # Fallback to default backend if V4L2 failed
        if prefer_v4l2:
            cap.release()
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                return False, None
        else:
            cap.release()
            return False, None

    # Try common settings to improve reliability
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warmup reads
    for _ in range(5):
        cap.read()
        time.sleep(0.01)

    ok, frame = cap.read()
    shape = None
    if ok and frame is not None:
        shape = (int(frame.shape[1]), int(frame.shape[0]))  # (w, h)

    cap.release()
    return bool(ok and frame is not None), shape


def probe_indices(max_indices: int, width: int, height: int, prefer_v4l2: bool) -> List[Tuple[int, Tuple[int, int]]]:
    usable: List[Tuple[int, Tuple[int, int]]] = []
    print(f"Probing camera indices 0..{max_indices - 1} with OpenCV {cv2.__version__}")
    for idx in range(max_indices):
        ok, shape = try_open_once(idx, width, height, prefer_v4l2)
        if ok and shape is not None:
            print(f"  [+] Camera index {idx}: OK, frame size {shape[0]}x{shape[1]}")
            usable.append((idx, shape))
        else:
            print(f"  [-] Camera index {idx}: not available")
    return usable


def dual_camera_test(indices: List[int], width: int, height: int, seconds: float, prefer_v4l2: bool, snapshot_dir: str) -> None:
    backend_flag = cv2.CAP_V4L2 if prefer_v4l2 else 0
    caps: List[cv2.VideoCapture] = []
    for idx in indices:
        cap = cv2.VideoCapture(idx, backend_flag) if backend_flag else cv2.VideoCapture(idx)
        if not cap.isOpened():
            # Fallback to default backend if V4L2 failed
            if prefer_v4l2:
                cap.release()
                cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            for c in caps:
                c.release()
            raise RuntimeError(f"Failed to open camera index {idx} for dual test")

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        caps.append(cap)

    print(f"\nStarting dual-camera read for {seconds:.1f}s on indices {indices}...")
    start = time.time()
    counts = [0 for _ in indices]
    saved_snapshot = [False for _ in indices]

    os.makedirs(snapshot_dir, exist_ok=True)

    while time.time() - start < seconds:
        for i, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok and frame is not None:
                counts[i] += 1
                if not saved_snapshot[i]:
                    out_path = os.path.join(snapshot_dir, f"opencv_cam_{indices[i]}.jpg")
                    try:
                        cv2.imwrite(out_path, frame)
                        print(f"  Saved snapshot for index {indices[i]} -> {out_path}")
                    except Exception as exc:
                        print(f"  Warning: failed to save snapshot for index {indices[i]}: {exc}")
                    saved_snapshot[i] = True
        time.sleep(0.005)

    for cap in caps:
        cap.release()

    print("Dual-camera read complete. Frame counts:")
    for idx, n in zip(indices, counts):
        print(f"  index {idx}: {n} frames")
    if all(n > 0 for n in counts):
        print("SUCCESS: Both cameras produced frames concurrently.")
    else:
        print("ERROR: One or both cameras produced zero frames during the test.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe and test dual cameras with OpenCV")
    parser.add_argument("--max-indices", type=int, default=8, help="Probe indices in range [0, max) (default: 8)")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height (default: 720)")
    parser.add_argument("--seconds", type=float, default=3.0, help="Duration for dual read test (default: 3.0s)")
    parser.add_argument("--no-v4l2", action="store_true", help="Do not prefer V4L2 backend on Linux")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    prefer_v4l2 = (sys.platform.startswith("linux") and not args.no_v4l2)

    symlinks = list_v4l_symlinks()
    if symlinks:
        print("Discovered video device symlinks:")
        for line in symlinks:
            print(f"  {line}")
        print()

    usable = probe_indices(args.max_indices, args.width, args.height, prefer_v4l2)
    if len(usable) < 2:
        print("\nFewer than two usable cameras detected. Connect both cameras and ensure no other program is using them.")
        return 1

    # Use the first two usable indices for the dual test
    indices = [usable[0][0], usable[1][0]]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_dir = os.path.join(script_dir)
    dual_camera_test(indices, args.width, args.height, args.seconds, prefer_v4l2, snapshot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


