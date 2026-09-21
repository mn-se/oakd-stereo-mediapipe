import argparse
from pathlib import Path
from queue import Empty, Full

import cv2
import depthai as dai
import numpy as np
import onnxruntime as ort
from stereo_common import (
    DISPLAY_IMAGE_WIDTH,
    HAND_CONNECTIONS,
    INFERENCE_HEIGHT,
    INFERENCE_WIDTH,
    MEASUREMENT_HEIGHT,
    MEASUREMENT_WIDTH,
    PointSmoother,
    StereoFrameSynchronizer as CommonStereoFrameSynchronizer,
    create_camera_pipeline as common_create_camera_pipeline,
    triangulate_landmarks as common_triangulate_landmarks,
)


PALM_INPUT_SIZE = 192
LANDMARK_INPUT_SIZE = 224
NUM_LANDMARKS = 21
MIN_LANDMARK_SPAN = 20.0

# Matches MediaPipe's HandLandmarksToRectCalculator: rotation/bbox are derived
# from this 12-point subset of the 21 hand landmarks (wrist, thumb CMC/MCP/IP,
# and the MCP/PIP joints of the other four fingers), not the full landmark set.
ROI_SUBSET_INDICES = np.asarray([0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18])
# Matches MediaPipe's RectTransformationCalculator options used for the
# landmark-derived hand rect (scale_x/y=2.0, shift_y=-0.1, square_long=true).
TRACKING_ROI_SCALE = 2.0
TRACKING_ROI_SHIFT_Y = -0.1

def generate_palm_anchors():
    anchors = []
    strides = [8, 16, 16, 16]
    layer = 0
    while layer < len(strides):
        end = layer
        while end < len(strides) and strides[end] == strides[layer]:
            end += 1
        cells = PALM_INPUT_SIZE // strides[layer]
        repeats = 2 * (end - layer)
        for y in range(cells):
            for x in range(cells):
                anchors.extend(
                    [(x + 0.5) / cells, (y + 0.5) / cells
                ] * repeats
                )
        layer = end
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 2)


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def weighted_nms(detections, threshold=0.3):
    remaining = detections[np.argsort(-detections[:, 0])]
    output = []
    while len(remaining):
        top = remaining[0]
        x1 = remaining[:, 1] - remaining[:, 3] / 2
        y1 = remaining[:, 2] - remaining[:, 4] / 2
        x2 = remaining[:, 1] + remaining[:, 3] / 2
        y2 = remaining[:, 2] + remaining[:, 4] / 2
        tx1 = top[1] - top[3] / 2
        ty1 = top[2] - top[4] / 2
        tx2 = top[1] + top[3] / 2
        ty2 = top[2] + top[4] / 2
        intersection = np.maximum(0, np.minimum(x2, tx2) - np.maximum(x1, tx1))
        intersection *= np.maximum(0, np.minimum(y2, ty2) - np.maximum(y1, ty1))
        union = (x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - intersection
        iou = intersection / np.maximum(union, 1e-9)
        overlapping = remaining[iou > threshold]
        weights = overlapping[:, :1]
        blended = top.copy()
        blended[1:] = (overlapping[:, 1:] * weights).sum(axis=0) / weights.sum()
        output.append(blended)
        remaining = remaining[iou <= threshold]
    return np.asarray(output, dtype=np.float32)


def warp_roi(image, roi, output_size):
    center_x, center_y, side, angle = roi
    matrix = cv2.getRotationMatrix2D(
        (center_x, center_y), angle, output_size / side
    )
    matrix[0, 2] += output_size / 2 - center_x
    matrix[1, 2] += output_size / 2 - center_y
    crop = cv2.warpAffine(image, matrix, (output_size, output_size))
    return crop, cv2.invertAffineTransform(matrix)


def palm_roi(palm):
    _, center_x, center_y, width, wrist_x, wrist_y, mcp_x, mcp_y = palm
    direction = np.asarray([mcp_x - wrist_x, mcp_y - wrist_y])
    length = np.linalg.norm(direction)
    unit = direction / length if length > 1e-6 else np.asarray([0.0, -1.0])
    center_x += 0.5 * width * unit[0]
    center_y += 0.5 * width * unit[1]
    angle = np.degrees(np.arctan2(unit[1], unit[0])) + 90.0
    return center_x, center_y, 2.6 * width, angle


def preprocess_rgb(image, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0


class OnnxHandTracker:
    def __init__(self, detector_path, landmark_path, threshold=0.5, tracking=False):
        self.detector = ort.InferenceSession(
            str(detector_path), providers=["CPUExecutionProvider"]
        )
        self.landmarker = ort.InferenceSession(
            str(landmark_path), providers=["CPUExecutionProvider"]
        )
        self.detector_input = self.detector.get_inputs()[0].name
        self.landmark_input = self.landmarker.get_inputs()[0].name
        self._validate_model_contract()
        self.anchors = generate_palm_anchors()
        self.threshold = threshold
        self.detector_threshold = threshold
        self.tracking = tracking
        self._previous_landmarks = None
        self._previous_roi = None
        if len(self.anchors) != 2016:
            raise RuntimeError(f"Unexpected palm anchor count: {len(self.anchors)}")

    def _validate_model_contract(self):
        detector_shape = self.detector.get_inputs()[0].shape
        landmark_shape = self.landmarker.get_inputs()[0].shape
        if detector_shape != [1, PALM_INPUT_SIZE, PALM_INPUT_SIZE, 3]:
            raise ValueError(
                "Unexpected palm detector input shape: "
                f"{detector_shape}"
            )
        if landmark_shape != [1, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3]:
            raise ValueError(
                "Unexpected landmark model input shape: "
                f"{landmark_shape}"
            )

        detector_outputs = self.detector.get_outputs()
        landmark_outputs = self.landmarker.get_outputs()
        if len(detector_outputs) != 2:
            raise ValueError(
                "Expected two palm detector outputs, got "
                f"{len(detector_outputs)}"
            )
        if len(landmark_outputs) < 2:
            raise ValueError(
                "Expected landmark and presence outputs, got "
                f"{len(landmark_outputs)}"
            )
        if detector_outputs[0].shape[-1] < 10:
            raise ValueError(
                "Palm detector regressors have too few values: "
                f"{detector_outputs[0].shape}"
            )
        if landmark_outputs[0].shape[-1] != NUM_LANDMARKS * 3:
            raise ValueError(
                "Unexpected landmark output shape: "
                f"{landmark_outputs[0].shape}"
            )

    def detect_palms(self, image):
        height, width = image.shape[:2]
        scale = PALM_INPUT_SIZE / max(height, width)
        pad_x = (PALM_INPUT_SIZE - width * scale) / 2
        pad_y = (PALM_INPUT_SIZE - height * scale) / 2
        matrix = np.asarray([[scale, 0, pad_x], [0, scale, pad_y]], dtype=np.float32)
        square = cv2.warpAffine(image, matrix, (PALM_INPUT_SIZE, PALM_INPUT_SIZE))
        input_image = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        raw_regressors, raw_scores = self.detector.run(
            None, {self.detector_input: input_image[None]}
        )
        regressors = raw_regressors[0]
        scores = sigmoid(raw_scores[0].reshape(-1))
        keep = scores >= self.detector_threshold
        if not np.any(keep):
            return []
        regressors = regressors[keep]
        anchors = self.anchors[keep]
        scores = scores[keep]
        rows = np.empty((len(scores), 9), dtype=np.float32)
        rows[:, 0] = scores
        rows[:, 1:3] = regressors[:, 0:2] / PALM_INPUT_SIZE + anchors
        rows[:, 3:5] = regressors[:, 2:4] / PALM_INPUT_SIZE
        rows[:, 5:7] = regressors[:, 4:6] / PALM_INPUT_SIZE + anchors
        rows[:, 7:9] = regressors[:, 8:10] / PALM_INPUT_SIZE + anchors
        rows = weighted_nms(rows)
        if len(rows) == 0:
            return []

        def unletterbox(points):
            return (points * PALM_INPUT_SIZE - [pad_x, pad_y]) / scale

        palms = []
        for row in rows:
            center = unletterbox(row[1:3])
            wrist = unletterbox(row[5:7])
            mcp = unletterbox(row[7:9])
            palms.append(
                np.asarray(
                    [
                        row[0],
                        center[0],
                        center[1],
                        max(row[3], row[4]) * PALM_INPUT_SIZE / scale,
                        wrist[0],
                        wrist[1],
                        mcp[0],
                        mcp[1],
                    ],
                    dtype=np.float32,
                )
            )
        return palms

    @staticmethod
    def tracking_roi(landmarks):
        # Replicates MediaPipe's HandLandmarksToRectCalculator followed by
        # RectTransformationCalculator(scale_x=scale_y=2.0, shift_y=-0.1,
        # square_long=true), instead of an ad-hoc bbox of all 21 landmarks.
        points = landmarks[ROI_SUBSET_INDICES, :2]
        wrist = points[0]
        # Weighted average of index/middle/ring MCP joints (subset positions
        # 4, 6, 8), matching the official rotation-target computation.
        target = (points[4] + points[8]) / 2.0
        target = (target + points[6]) / 2.0
        direction = target - wrist
        length = np.linalg.norm(direction)
        unit = direction / length if length > 1e-6 else np.asarray([0.0, -1.0])
        rotation = np.arctan2(unit[1], unit[0]) + np.pi / 2.0

        # Axis-aligned center of the subset, then bbox in the rotated frame.
        axis_center = (points.max(axis=0) + points.min(axis=0)) / 2.0
        cos_r, sin_r = np.cos(-rotation), np.sin(-rotation)
        rel = points - axis_center
        projected_x = rel[:, 0] * cos_r - rel[:, 1] * sin_r
        projected_y = rel[:, 0] * sin_r + rel[:, 1] * cos_r
        proj_min = np.asarray([projected_x.min(), projected_y.min()])
        proj_max = np.asarray([projected_x.max(), projected_y.max()])
        proj_center = (proj_min + proj_max) / 2.0
        width, height = (proj_max - proj_min).tolist()

        cos_f, sin_f = np.cos(rotation), np.sin(rotation)
        center = np.asarray(
            [
                proj_center[0] * cos_f - proj_center[1] * sin_f + axis_center[0],
                proj_center[0] * sin_f + proj_center[1] * cos_f + axis_center[1],
            ]
        )

        # RectTransformationCalculator applies the shift (using the
        # pre-square width/height) before squaring and scaling.
        center = center + np.asarray(
            [
                -height * TRACKING_ROI_SHIFT_Y * sin_f,
                height * TRACKING_ROI_SHIFT_Y * cos_f,
            ]
        )
        side = max(max(width, height) * TRACKING_ROI_SCALE, 1.0)
        angle = np.degrees(rotation)
        return float(center[0]), float(center[1]), float(side), float(angle)

    def _run_landmark_model(self, image, roi):
        crop, inverse = warp_roi(image, roi, LANDMARK_INPUT_SIZE)
        input_image = preprocess_rgb(crop, LANDMARK_INPUT_SIZE)
        outputs = self.landmarker.run(
            None, {self.landmark_input: input_image[None]}
        )
        screen = outputs[0][0].reshape(NUM_LANDMARKS, 3).astype(np.float32)
        presence = float(np.clip(outputs[1][0][0], 0.0, 1.0))
        if presence < self.threshold:
            return None, presence
        screen[:, :2] = screen[:, :2] @ inverse[:, :2].T + inverse[:, 2]
        screen[:, 2] *= roi[2] / LANDMARK_INPUT_SIZE
        return screen, presence

    @staticmethod
    def _valid_landmarks(landmarks, image_shape):
        if landmarks is None or not np.isfinite(landmarks).all():
            return False

        height, width = image_shape[:2]
        points = landmarks[:, :2]
        span = points.max(axis=0) - points.min(axis=0)
        inside = (
            (points[:, 0] >= -0.25 * width)
            & (points[:, 0] <= 1.25 * width)
            & (points[:, 1] >= -0.25 * height)
            & (points[:, 1] <= 1.25 * height)
        )
        return bool(
            span.max() >= MIN_LANDMARK_SPAN
            and inside.mean() >= 0.8
        )

    @staticmethod
    def _consistent_tracking(previous, current):
        previous_points = previous[:, :2]
        current_points = current[:, :2]
        previous_span = previous_points.max(axis=0) - previous_points.min(axis=0)
        current_span = current_points.max(axis=0) - current_points.min(axis=0)
        span_ratio = current_span.max() / max(previous_span.max(), 1.0)
        displacement = np.linalg.norm(current_points[0] - previous_points[0])
        return bool(
            0.5 <= span_ratio <= 2.0
            and displacement <= max(previous_span.max() * 0.75, 40.0)
        )

    def _detect_from_palm(self, image):
        palms = self.detect_palms(image)
        if not palms:
            return None, None

        detected_roi = palm_roi(palms[0])
        detected, _ = self._run_landmark_model(image, detected_roi)
        if self._valid_landmarks(detected, image.shape):
            return detected, detected_roi
        return None, None

    def detect(self, image):
        if not self.tracking:
            self._previous_landmarks = None
            self._previous_roi = None
            detected, _ = self._detect_from_palm(image)
            return detected

        if (
            self._previous_landmarks is not None
            and self._previous_roi is not None
        ):
            previous_landmarks = self._previous_landmarks
            tracking_roi = self.tracking_roi(previous_landmarks)
            tracked, _ = self._run_landmark_model(image, tracking_roi)
            if (
                self._valid_landmarks(tracked, image.shape)
                and self._consistent_tracking(previous_landmarks, tracked)
            ):
                self._previous_landmarks = tracked
                self._previous_roi = self.tracking_roi(tracked)
                return tracked
            self._previous_landmarks = None
            self._previous_roi = None

        detected, detected_roi = self._detect_from_palm(image)
        if detected is not None:
            self._previous_landmarks = detected
            self._previous_roi = detected_roi
            return detected
        self._previous_landmarks = None
        self._previous_roi = None
        return None


def triangulate(left_landmarks, right_landmarks, p_left, p_right):
    if left_landmarks is None or right_landmarks is None:
        return None
    return common_triangulate_landmarks(
        left_landmarks,
        right_landmarks,
        p_left,
        p_right,
    )


def scale_landmarks(landmarks):
    if landmarks is None:
        return None

    scaled = landmarks.copy()
    scaled[:, 0] *= MEASUREMENT_WIDTH / INFERENCE_WIDTH
    scaled[:, 1] *= MEASUREMENT_HEIGHT / INFERENCE_HEIGHT
    scaled[:, 2] *= MEASUREMENT_WIDTH / INFERENCE_WIDTH
    return scaled


def draw_landmarks(image, landmarks):
    points = []
    for landmark in landmarks:
        point = (int(landmark[0]), int(landmark[1]))
        points.append(point)
        cv2.circle(image, point, 4, (0, 255, 0), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], (0, 200, 255), 2)


def draw_coordinate_label(image, landmark, point_3d_mm):
    height, width = image.shape[:2]
    x = min(width - 260, max(6, int(landmark[0])))
    y = min(height - 10, max(34, int(landmark[1])))
    x_mm, y_mm, z_mm = point_3d_mm
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


def parse_args():
    parser = argparse.ArgumentParser(description="OAK-D stereo hand 3D with ONNX Runtime.")
    parser.add_argument("--detector", type=Path, default=Path("models/hand_detector.onnx"))
    parser.add_argument("--landmarks", type=Path, default=Path("models/hand_landmarks_detector.onnx"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Run Palm Detection on every frame instead of tracking.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tracker_left = OnnxHandTracker(
        args.detector,
        args.landmarks,
        args.threshold,
        not args.no_tracking,
    )
    tracker_right = OnnxHandTracker(
        args.detector,
        args.landmarks,
        args.threshold,
        not args.no_tracking,
    )
    device = dai.Device()
    pipeline, left_queue, right_queue, rectification = common_create_camera_pipeline(device)
    map_left_x, map_left_y, map_right_x, map_right_y, p_left, p_right = rectification
    synchronizer = CommonStereoFrameSynchronizer(left_queue, right_queue)
    point_smoother = PointSmoother()

    try:
        while pipeline.isRunning():
            pair = synchronizer.poll()
            if pair is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            left_frame, right_frame = pair
            left = cv2.remap(left_frame.getCvFrame(), map_left_x, map_left_y, cv2.INTER_LINEAR)
            right = cv2.remap(right_frame.getCvFrame(), map_right_x, map_right_y, cv2.INTER_LINEAR)
            inference_left = cv2.resize(
                left,
                (INFERENCE_WIDTH, INFERENCE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            inference_right = cv2.resize(
                right,
                (INFERENCE_WIDTH, INFERENCE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            inference_landmarks_left = tracker_left.detect(inference_left)
            inference_landmarks_right = tracker_right.detect(inference_right)
            left_landmarks = scale_landmarks(inference_landmarks_left)
            right_landmarks = scale_landmarks(inference_landmarks_right)
            raw_points_3d = triangulate(
                left_landmarks,
                right_landmarks,
                p_left,
                p_right,
            )
            points_3d = point_smoother.update(raw_points_3d)
            if points_3d is not None:
                draw_landmarks(left, left_landmarks)
                draw_landmarks(right, right_landmarks)
                index_tip = points_3d[8]
                print(f"\rIndex tip: X={index_tip[0]:7.1f} Y={index_tip[1]:7.1f} Z={index_tip[2]:7.1f} mm", end="", flush=True)
            scale = DISPLAY_IMAGE_WIDTH / MEASUREMENT_WIDTH
            display_height = int(MEASUREMENT_HEIGHT * scale)
            display_left = cv2.resize(
                left,
                (DISPLAY_IMAGE_WIDTH, display_height),
                interpolation=cv2.INTER_AREA,
            )
            display_right = cv2.resize(
                right,
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
            if points_3d is not None:
                draw_coordinate_label(
                    display_left,
                    left_landmarks[8] * scale,
                    points_3d[8],
                )
                draw_coordinate_label(
                    display_right,
                    right_landmarks[8] * scale,
                    points_3d[8],
                )
            combined = np.hstack((display_left, display_right))
            cv2.imshow("OAK-D Stereo Hand 3D (ONNX)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
