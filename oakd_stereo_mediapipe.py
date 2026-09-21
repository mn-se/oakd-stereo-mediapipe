import argparse
import multiprocessing
from queue import Empty, Full
from pathlib import Path
from typing import NamedTuple

import cv2
import depthai as dai
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MEASUREMENT_WIDTH = 1280
MEASUREMENT_HEIGHT = 800
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 400
DISPLAY_IMAGE_WIDTH = 640


# MediaPipe Hand landmark IDs
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


class LandmarkPoint(NamedTuple):
    x: float
    y: float
    z: float


class StereoFrameSynchronizer:
    def __init__(self, left_queue, right_queue):
        self._left_queue = left_queue
        self._right_queue = right_queue
        self._left_frames = {}
        self._right_frames = {}

    def poll(self):
        self._read_frame(self._left_queue, self._left_frames)
        self._read_frame(self._right_queue, self._right_frames)

        common_sequences = (
            self._left_frames.keys() & self._right_frames.keys()
        )
        if not common_sequences:
            self._discard_old_frames()
            return None

        sequence = max(common_sequences)
        left_frame = self._left_frames.pop(sequence)
        right_frame = self._right_frames.pop(sequence)
        self._discard_through(sequence)
        return left_frame, right_frame

    @staticmethod
    def _read_frame(queue, frames):
        frame = queue.tryGet()
        if frame is not None:
            frames[frame.getSequenceNum()] = frame

    def _discard_old_frames(self):
        if not self._left_frames or not self._right_frames:
            return

        oldest_available = max(
            min(self._left_frames),
            min(self._right_frames),
        )
        self._discard_through(oldest_available - 1)

    def _discard_through(self, sequence):
        for frames in (self._left_frames, self._right_frames):
            for frame_sequence in list(frames):
                if frame_sequence <= sequence:
                    del frames[frame_sequence]


class PointSmoother:
    def __init__(self, alpha=0.25):
        self._alpha = alpha
        self._value = None

    def update(self, value):
        if value is None:
            self._value = None
            return None

        value = np.asarray(value, dtype=np.float64)
        if self._value is None:
            self._value = value
        else:
            self._value = (
                self._alpha * value
                + (1.0 - self._alpha) * self._value
            )
        return self._value.copy()


def create_rectification_maps(calib):
    """Create stereo rectification maps from OAK-D factory calibration."""

    camera_left = dai.CameraBoardSocket.CAM_B
    camera_right = dai.CameraBoardSocket.CAM_C

    k_left = np.array(
        calib.getCameraIntrinsics(
            camera_left,
            MEASUREMENT_WIDTH,
            MEASUREMENT_HEIGHT,
        ),
        dtype=np.float64,
    )

    k_right = np.array(
        calib.getCameraIntrinsics(
            camera_right,
            MEASUREMENT_WIDTH,
            MEASUREMENT_HEIGHT,
        ),
        dtype=np.float64,
    )

    d_left = np.array(
        calib.getDistortionCoefficients(camera_left),
        dtype=np.float64,
    )

    d_right = np.array(
        calib.getDistortionCoefficients(camera_right),
        dtype=np.float64,
    )

    extrinsics = np.array(
        calib.getCameraExtrinsics(
            camera_left,
            camera_right,
        ),
        dtype=np.float64,
    )

    r = extrinsics[:3, :3]
    t = extrinsics[:3, 3].reshape(3, 1)

    image_size = (MEASUREMENT_WIDTH, MEASUREMENT_HEIGHT)

    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        r,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        k_left,
        d_left,
        r1,
        p1,
        image_size,
        cv2.CV_32FC1,
    )

    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        k_right,
        d_right,
        r2,
        p2,
        image_size,
        cv2.CV_32FC1,
    )

    return (
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        p1,
        p2,
        _,
        extrinsics,
    )


def triangulate_landmarks(
    landmarks_left,
    landmarks_right,
    p_left,
    p_right,
    image_width,
    image_height,
):
    """Triangulate corresponding MediaPipe landmarks."""

    points_left = []
    points_right = []

    for landmark_left, landmark_right in zip(landmarks_left, landmarks_right):
        x_left = landmark_left.x * image_width
        y_left = landmark_left.y * image_height

        x_right = landmark_right.x * image_width
        y_right = landmark_right.y * image_height

        points_left.append([x_left, y_left])
        points_right.append([x_right, y_right])

    points_left = np.asarray(points_left, dtype=np.float64).T
    points_right = np.asarray(points_right, dtype=np.float64).T

    points_4d = cv2.triangulatePoints(
        p_left,
        p_right,
        points_left,
        points_right,
    )

    points_3d = points_4d[:3] / points_4d[3]

    # P matrices use cm because OAK-D extrinsics are in cm.
    # Convert the result to mm.
    points_3d_mm = points_3d.T * 10.0

    return points_3d_mm


def draw_landmarks(image, landmarks):
    """Draw MediaPipe landmarks."""

    height, width = image.shape[:2]

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ]

    pixel_points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        pixel_points.append((x, y))

        cv2.circle(
            image,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )

    for index_a, index_b in connections:
        cv2.line(
            image,
            pixel_points[index_a],
            pixel_points[index_b],
            (0, 200, 255),
            2,
        )

def draw_coordinate_labels(image, landmarks, points_3d_mm):
    """Draw readable 3D coordinates on the resized display image."""

    height, width = image.shape[:2]
    for index in (8,):
        landmark = landmarks[index]
        x = min(width - 260, max(6, int(landmark.x * width)))
        y = min(height - 10, max(34, int(landmark.y * height)))
        x_mm, y_mm, z_mm = points_3d_mm[index]
        text = f"X:{x_mm:.0f} Y:{y_mm:.0f} Z:{z_mm:.0f} mm"

        cv2.putText(
            image,
            text,
            (x + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def create_hand_landmarker(model_path):
    """Create a MediaPipe Tasks hand landmarker for video frames."""

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


class HandInferenceWorker:
    def __init__(self, model_path):
        context = multiprocessing.get_context("spawn")
        self._frames = context.Queue(maxsize=1)
        self._results = context.Queue(maxsize=1)
        self._model_path = model_path
        self._process = context.Process(
            target=HandInferenceWorker._run,
            args=(self._frames, self._results, self._model_path),
            name="hand-inference",
            daemon=True,
        )
        self._process.start()

    def submit(self, rgb_left, rgb_right):
        try:
            self._frames.get_nowait()
        except Empty:
            pass

        try:
            self._frames.put_nowait((rgb_left, rgb_right))
        except Full:
            pass

    def poll(self):
        latest_result = None
        while True:
            try:
                latest_result = self._results.get_nowait()
            except Empty:
                return latest_result

    @staticmethod
    def _run(frames, results, model_path):
        hands_left = create_hand_landmarker(model_path)
        hands_right = create_hand_landmarker(model_path)
        timestamp_ms = 0

        try:
            while True:
                images = frames.get()
                if images is None:
                    break

                rgb_left, rgb_right = images
                timestamp_ms += 33
                image_left = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_left,
                )
                image_right = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_right,
                )
                result_left = hands_left.detect_for_video(
                    image_left,
                    timestamp_ms,
                )
                result_right = hands_right.detect_for_video(
                    image_right,
                    timestamp_ms,
                )

                landmarks_left = (
                    [
                        LandmarkPoint(
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        )
                        for landmark in result_left.hand_landmarks[0]
                    ]
                    if result_left.hand_landmarks
                    else None
                )
                landmarks_right = (
                    [
                        LandmarkPoint(
                            landmark.x,
                            landmark.y,
                            landmark.z,
                        )
                        for landmark in result_right.hand_landmarks[0]
                    ]
                    if result_right.hand_landmarks
                    else None
                )

                while True:
                    try:
                        results.get_nowait()
                    except Empty:
                        break

                try:
                    results.put_nowait((landmarks_left, landmarks_right))
                except Full:
                    pass
        finally:
            hands_left.close()
            hands_right.close()

    def close(self):
        try:
            self._frames.get_nowait()
        except Empty:
            pass

        try:
            self._frames.put_nowait(None)
        except Full:
            pass

        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._frames.close()
        self._results.close()


def create_camera_pipeline(device):
    calib = device.readCalibration()

    rectification = create_rectification_maps(calib)
    (
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        p_left,
        p_right,
        _,
        extrinsics,
    ) = rectification

    print("=== Rectified P1 ===")
    print(p_left)
    print()
    print("=== Rectified P2 ===")
    print(p_right)
    print()
    print("=== Left -> Right Extrinsics [cm] ===")
    print(extrinsics)

    baseline_cm = calib.getBaselineDistance(
        dai.CameraBoardSocket.CAM_B,
        dai.CameraBoardSocket.CAM_C,
    )
    print()
    print("=== Baseline ===")
    print(f"{baseline_cm:.3f} cm")
    print(f"{baseline_cm * 10.0:.1f} mm")

    pipeline = dai.Pipeline(device)
    left = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B
    )
    right = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_C
    )

    left_queue = left.requestOutput(
        (MEASUREMENT_WIDTH, MEASUREMENT_HEIGHT),
        dai.ImgFrame.Type.GRAY8,
    ).createOutputQueue(
        maxSize=4,
        blocking=False,
    )
    right_queue = right.requestOutput(
        (MEASUREMENT_WIDTH, MEASUREMENT_HEIGHT),
        dai.ImgFrame.Type.GRAY8,
    ).createOutputQueue(
        maxSize=4,
        blocking=False,
    )

    pipeline.start()
    return pipeline, left_queue, right_queue, rectification


def prepare_stereo_frames(frame_left, frame_right, rectification):
    (
        map_left_x,
        map_left_y,
        map_right_x,
        map_right_y,
        _,
        _,
        _,
        _,
    ) = rectification

    rect_left = cv2.remap(
        frame_left,
        map_left_x,
        map_left_y,
        cv2.INTER_LINEAR,
    )
    rect_right = cv2.remap(
        frame_right,
        map_right_x,
        map_right_y,
        cv2.INTER_LINEAR,
    )
    inference_left = cv2.resize(
        rect_left,
        (INFERENCE_WIDTH, INFERENCE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    inference_right = cv2.resize(
        rect_right,
        (INFERENCE_WIDTH, INFERENCE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb_left = cv2.cvtColor(inference_left, cv2.COLOR_GRAY2RGB)
    rgb_right = cv2.cvtColor(inference_right, cv2.COLOR_GRAY2RGB)

    return rect_left, rect_right, rgb_left, rgb_right


def calculate_points_3d(landmarks_left, landmarks_right, p_left, p_right):
    if landmarks_left is None or landmarks_right is None:
        return None

    points_3d_mm = triangulate_landmarks(
        landmarks_left,
        landmarks_right,
        p_left,
        p_right,
        MEASUREMENT_WIDTH,
        MEASUREMENT_HEIGHT,
    )


    return points_3d_mm


def print_landmark_coordinates(points_3d_mm):
    index_tip = points_3d_mm[8]
    wrist = points_3d_mm[0]
    print(
        f"\r"
        f"Index tip: "
        f"X={index_tip[0]:7.1f} "
        f"Y={index_tip[1]:7.1f} "
        f"Z={index_tip[2]:7.1f} mm    "
        f"Wrist: "
        f"X={wrist[0]:7.1f} "
        f"Y={wrist[1]:7.1f} "
        f"Z={wrist[2]:7.1f} mm",
        end="",
        flush=True,
    )


def build_display_frame(
    rect_left,
    rect_right,
    landmarks_left,
    landmarks_right,
    points_3d_mm,
):
    if points_3d_mm is not None:
        draw_landmarks(rect_left, landmarks_left)
        draw_landmarks(rect_right, landmarks_right)

    display_scale = DISPLAY_IMAGE_WIDTH / MEASUREMENT_WIDTH
    display_height = int(MEASUREMENT_HEIGHT * display_scale)
    display_left = cv2.resize(
        rect_left,
        (DISPLAY_IMAGE_WIDTH, display_height),
        interpolation=cv2.INTER_AREA,
    )
    display_right = cv2.resize(
        rect_right,
        (DISPLAY_IMAGE_WIDTH, display_height),
        interpolation=cv2.INTER_AREA,
    )

    cv2.putText(
        display_left,
        "LEFT",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_right,
        "RIGHT",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if points_3d_mm is not None:
        draw_coordinate_labels(
            display_left,
            landmarks_left,
            points_3d_mm,
        )
        draw_coordinate_labels(
            display_right,
            landmarks_right,
            points_3d_mm,
        )

    return np.hstack((display_left, display_right))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate 3D hand landmarks from an OAK-D stereo pair."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hand_landmarker.task"),
        help="Path to the MediaPipe Hand Landmarker .task model.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe model not found: {model_path}. "
            "Download hand_landmarker.task and pass it with --model."
        )

    device = dai.Device()
    pipeline, left_queue, right_queue, rectification = create_camera_pipeline(
        device
    )
    _, _, _, _, p_left, p_right, _, _ = rectification

    inference_worker = HandInferenceWorker(model_path)
    frame_synchronizer = StereoFrameSynchronizer(
        left_queue,
        right_queue,
    )
    point_smoother = PointSmoother()
    latest_landmarks_left = None
    latest_landmarks_right = None

    try:
        while pipeline.isRunning():
            synchronized_frames = frame_synchronizer.poll()

            if synchronized_frames is None:
                if cv2.waitKey(1) == ord("q"):
                    break
                continue

            left_frame, right_frame = synchronized_frames

            frame_left = left_frame.getCvFrame()
            frame_right = right_frame.getCvFrame()

            rect_left, rect_right, rgb_left, rgb_right = prepare_stereo_frames(
                frame_left,
                frame_right,
                rectification,
            )

            inference_worker.submit(rgb_left, rgb_right)
            inference_result = inference_worker.poll()

            if inference_result is not None:
                latest_landmarks_left, latest_landmarks_right = inference_result

            raw_points_3d_mm = calculate_points_3d(
                latest_landmarks_left,
                latest_landmarks_right,
                p_left,
                p_right,
            )
            points_3d_mm = point_smoother.update(raw_points_3d_mm)
            if points_3d_mm is not None:
                print_landmark_coordinates(points_3d_mm)

            combined = build_display_frame(
                rect_left,
                rect_right,
                latest_landmarks_left,
                latest_landmarks_right,
                points_3d_mm,
            )

            cv2.imshow("OAK-D Stereo Hand 3D", combined)

            if cv2.waitKey(1) == ord("q"):
                break

    finally:
        inference_worker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()