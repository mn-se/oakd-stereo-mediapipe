from contextlib import contextmanager
import os
import threading

import cv2
import depthai as dai
import numpy as np


MEASUREMENT_WIDTH = 1280
MEASUREMENT_HEIGHT = 800
INFERENCE_WIDTH = 640
INFERENCE_HEIGHT = 400
DISPLAY_IMAGE_WIDTH = 640

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_NATIVE_STDERR_LOCK = threading.Lock()


@contextmanager
def suppress_native_stderr():
    with _NATIVE_STDERR_LOCK:
        saved_fd = os.dup(2)
        null_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_fd, 2)
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(null_fd)
            os.close(saved_fd)


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


def create_rectification_maps(calibration):
    camera_left = dai.CameraBoardSocket.CAM_B
    camera_right = dai.CameraBoardSocket.CAM_C
    image_size = (MEASUREMENT_WIDTH, MEASUREMENT_HEIGHT)

    k_left = np.asarray(
        calibration.getCameraIntrinsics(
            camera_left,
            MEASUREMENT_WIDTH,
            MEASUREMENT_HEIGHT,
        ),
        dtype=np.float64,
    )
    k_right = np.asarray(
        calibration.getCameraIntrinsics(
            camera_right,
            MEASUREMENT_WIDTH,
            MEASUREMENT_HEIGHT,
        ),
        dtype=np.float64,
    )
    d_left = np.asarray(
        calibration.getDistortionCoefficients(camera_left),
        dtype=np.float64,
    )
    d_right = np.asarray(
        calibration.getDistortionCoefficients(camera_right),
        dtype=np.float64,
    )
    extrinsics = np.asarray(
        calibration.getCameraExtrinsics(camera_left, camera_right),
        dtype=np.float64,
    )

    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3].reshape(3, 1)
    r_left, r_right, p_left, p_right, _, _, _ = cv2.stereoRectify(
        k_left,
        d_left,
        k_right,
        d_right,
        image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map_left = cv2.initUndistortRectifyMap(
        k_left,
        d_left,
        r_left,
        p_left,
        image_size,
        cv2.CV_32FC1,
    )
    map_right = cv2.initUndistortRectifyMap(
        k_right,
        d_right,
        r_right,
        p_right,
        image_size,
        cv2.CV_32FC1,
    )
    return map_left[0], map_left[1], map_right[0], map_right[1], p_left, p_right


def create_camera_pipeline(device):
    calibration = device.readCalibration()
    rectification = create_rectification_maps(calibration)
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
    ).createOutputQueue(maxSize=4, blocking=False)
    right_queue = right.requestOutput(
        (MEASUREMENT_WIDTH, MEASUREMENT_HEIGHT),
        dai.ImgFrame.Type.GRAY8,
    ).createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    return pipeline, left_queue, right_queue, rectification


def triangulate_landmarks(landmarks_left, landmarks_right, p_left, p_right):
    if landmarks_left is None or landmarks_right is None:
        return None

    left = np.asarray(
        [[landmark[0], landmark[1]] for landmark in landmarks_left],
        dtype=np.float64,
    ).T
    right = np.asarray(
        [[landmark[0], landmark[1]] for landmark in landmarks_right],
        dtype=np.float64,
    ).T
    points_4d = cv2.triangulatePoints(p_left, p_right, left, right)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T * 10.0


def rectify_frames(frame_left, frame_right, rectification):
    map_left_x, map_left_y, map_right_x, map_right_y, _, _ = rectification
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
    return rect_left, rect_right
