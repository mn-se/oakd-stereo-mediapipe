import argparse
import os
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import mediapipe as mp
import numpy as np
from absl import logging as absl_logging
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from oakd_stereo_onnx import OnnxHandTracker, palm_roi, warp_roi
from stereo_common import (
    INFERENCE_HEIGHT,
    INFERENCE_WIDTH,
    MEASUREMENT_HEIGHT,
    MEASUREMENT_WIDTH,
    StereoFrameSynchronizer,
    create_camera_pipeline,
    rectify_frames,
    suppress_native_stderr,
)

absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold(absl_logging.ERROR)


LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def create_mediapipe_landmarker(model_path, running_mode=vision.RunningMode.IMAGE):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with suppress_native_stderr():
        return vision.HandLandmarker.create_from_options(options)


def detect_mediapipe(landmarker, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    with suppress_native_stderr():
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
    if not result.hand_landmarks:
        return None
    return np.asarray(
        [
            [
                landmark.x * MEASUREMENT_WIDTH,
                landmark.y * MEASUREMENT_HEIGHT,
                landmark.z,
            ]
            for landmark in result.hand_landmarks[0]
        ],
        dtype=np.float64,
    )


def detect_mediapipe_video(landmarker, image, timestamp_ms):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    with suppress_native_stderr():
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            timestamp_ms,
        )
    if not result.hand_landmarks:
        return None
    return np.asarray(
        [
            [
                landmark.x * MEASUREMENT_WIDTH,
                landmark.y * MEASUREMENT_HEIGHT,
                landmark.z,
            ]
            for landmark in result.hand_landmarks[0]
        ],
        dtype=np.float64,
    )


def detect_mediapipe_on_onnx_roi(landmarker, image, tracker):
    palms = tracker.detect_palms(image)
    if not palms:
        return None

    roi = palm_roi(palms[0])
    crop, inverse = warp_roi(image, roi, 224)
    result = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB),
        )
    )
    if not result.hand_landmarks:
        return None

    points = np.asarray(
        [
            [landmark.x * 224.0, landmark.y * 224.0, landmark.z]
            for landmark in result.hand_landmarks[0]
        ],
        dtype=np.float64,
    )
    points[:, :2] = points[:, :2] @ inverse[:, :2].T + inverse[:, 2]
    return points


def detect_onnx(tracker, image):
    landmarks = tracker.detect(image)
    if landmarks is None:
        return None
    scaled = landmarks.astype(np.float64).copy()
    scaled[:, 0] *= MEASUREMENT_WIDTH / INFERENCE_WIDTH
    scaled[:, 1] *= MEASUREMENT_HEIGHT / INFERENCE_HEIGHT
    return scaled


def inspect_onnx_pipeline(tracker, image, name, output_dir):
    palms = tracker.detect_palms(image)
    print(f"{name} ONNX palms: {len(palms)}")
    if not palms:
        return

    roi = palm_roi(palms[0])
    print(
        f"{name} ONNX palm: "
        f"score={palms[0][0]:.4f} "
        f"center=({palms[0][1]:.1f},{palms[0][2]:.1f}) "
        f"width={palms[0][3]:.1f}"
    )
    print(
        f"{name} ONNX ROI: "
        f"center=({roi[0]:.1f},{roi[1]:.1f}) "
        f"side={roi[2]:.1f} angle={roi[3]:.1f}"
    )
    crop, inverse = warp_roi(image, roi, 224)
    cv2.imwrite(str(output_dir / f"{name.lower()}_onnx_roi.png"), crop)
    input_image = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    raw_outputs = tracker.landmarker.run(
        None,
        {tracker.landmark_input: input_image[None]},
    )
    raw_landmarks = raw_outputs[0][0].reshape(21, 3)
    print(
        f"{name} ONNX raw landmarks: "
        f"x=[{raw_landmarks[:, 0].min():.1f},"
        f"{raw_landmarks[:, 0].max():.1f}] "
        f"y=[{raw_landmarks[:, 1].min():.1f},"
        f"{raw_landmarks[:, 1].max():.1f}]"
    )


def compare_tflite_onnx_landmark(task_path, tracker, image, name):
    try:
        import tensorflow as tf
    except ImportError:
        print("TFLite comparison skipped: install the conversion extra")
        return

    palms = tracker.detect_palms(image)
    if not palms:
        return

    crop, _ = warp_roi(image, palm_roi(palms[0]), 224)
    input_data = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    with tempfile.TemporaryDirectory() as temporary_dir:
        with ZipFile(task_path) as archive:
            archive.extract("hand_landmarks_detector.tflite", temporary_dir)
        tflite_path = Path(temporary_dir) / "hand_landmarks_detector.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], input_data[None])
        interpreter.invoke()
        tflite_outputs = [
            interpreter.get_tensor(detail["index"])
            for detail in interpreter.get_output_details()
        ]

    onnx_outputs = tracker.landmarker.run(
        None,
        {tracker.landmark_input: input_data[None]},
    )
    print(f"=== {name} TFLite/ONNX raw output comparison ===")
    for tflite_output in tflite_outputs:
        matches = [
            output
            for output in onnx_outputs
            if output.shape == tflite_output.shape
        ]
        if not matches:
            print(f"TFLite shape={tflite_output.shape}: no ONNX shape match")
            continue
        errors = [
            float(np.max(np.abs(tflite_output - output)))
            for output in matches
        ]
        print(
            f"shape={tflite_output.shape} "
            f"min_abs_diff={min(errors):.6g}"
        )


def compare_tflite_onnx_detector(task_path, tracker, image, name):
    try:
        import tensorflow as tf
    except ImportError:
        print("TFLite comparison skipped: install the conversion extra")
        return

    height, width = image.shape[:2]
    size = 192
    scale = size / max(height, width)
    pad_x = (size - width * scale) / 2
    pad_y = (size - height * scale) / 2
    matrix = np.asarray([[scale, 0, pad_x], [0, scale, pad_y]], dtype=np.float32)
    square = cv2.warpAffine(image, matrix, (size, size))
    input_data = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    with tempfile.TemporaryDirectory() as temporary_dir:
        with ZipFile(task_path) as archive:
            archive.extract("hand_detector.tflite", temporary_dir)
        tflite_path = Path(temporary_dir) / "hand_detector.tflite"
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_detail["index"], input_data[None])
        interpreter.invoke()
        tflite_outputs = [
            interpreter.get_tensor(detail["index"])
            for detail in interpreter.get_output_details()
        ]

    onnx_outputs = tracker.detector.run(
        None,
        {tracker.detector_input: input_data[None]},
    )
    print(f"=== {name} TFLite/ONNX detector raw output comparison ===")
    for tflite_output in tflite_outputs:
        matches = [
            output
            for output in onnx_outputs
            if output.shape == tflite_output.shape
        ]
        if not matches:
            print(f"TFLite shape={tflite_output.shape}: no ONNX shape match")
            continue
        errors = [
            float(np.max(np.abs(tflite_output - output)))
            for output in matches
        ]
        print(
            f"shape={tflite_output.shape} "
            f"min_abs_diff={min(errors):.6g}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MediaPipe and ONNX landmarks on one OAK-D stereo frame."
    )
    parser.add_argument("--model", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--detector", type=Path, default=Path("models/hand_detector.onnx"))
    parser.add_argument("--landmarks", type=Path, default=Path("models/hand_landmarks_detector.onnx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("comparison_debug"),
        help="Directory for captured frames and ROI crops.",
    )
    parser.add_argument(
        "--tracking-frames",
        type=int,
        default=0,
        help="If >0, also run a live tracking-mode sequence comparison "
        "(MediaPipe VIDEO mode vs ONNX tracking=True) for this many frames "
        "per side, to check for ROI drift over time. Keep the hand in view.",
    )
    return parser.parse_args()


def compare(name, mediapipe_points, onnx_points):
    print(f"=== {name} ===")
    if mediapipe_points is None or onnx_points is None:
        print(f"MediaPipe detected: {mediapipe_points is not None}")
        print(f"ONNX detected: {onnx_points is not None}")
        return

    differences = np.linalg.norm(
        mediapipe_points[:, :2] - onnx_points[:, :2],
        axis=1,
    )
    print(f"mean_xy_error_px: {differences.mean():.2f}")
    print(f"max_xy_error_px: {differences.max():.2f}")
    for index, error in enumerate(differences):
        print(
            f"{index:02d} {LANDMARK_NAMES[index]:22s} "
            f"error={error:7.2f}px "
            f"mp=({mediapipe_points[index, 0]:7.1f},"
            f"{mediapipe_points[index, 1]:7.1f}) "
            f"onnx=({onnx_points[index, 0]:7.1f},"
            f"{onnx_points[index, 1]:7.1f})"
        )


def compare_roi_outputs(name, mediapipe_points, onnx_points):
    print(f"=== {name} SAME ONNX ROI ===")
    if mediapipe_points is None or onnx_points is None:
        print(f"MediaPipe detected: {mediapipe_points is not None}")
        print(f"ONNX detected: {onnx_points is not None}")
        return
    differences = np.linalg.norm(
        mediapipe_points[:, :2] - onnx_points[:, :2],
        axis=1,
    )
    print(f"mean_xy_error_px: {differences.mean():.2f}")
    print(f"max_xy_error_px: {differences.max():.2f}")


def compare_tracking_sequence(
    mediapipe_left,
    mediapipe_right,
    onnx_left,
    onnx_right,
    get_frame_pair,
    num_frames,
    output_dir,
):
    """Runs both backends in their tracking mode (MediaPipe VIDEO mode with
    internal tracking, ONNX with tracking=True) over a live sequence of
    stereo frames, to check whether ONNX's tracking ROI drifts relative to
    MediaPipe over time. Requires the hand to stay roughly in view.

    Saves a debug frame + palm-candidate count the first time each side
    fails to detect on both backends, to help diagnose why (e.g. hand out
    of frame on one camera)."""
    print(f"=== TRACKING SEQUENCE ({num_frames} frames) ===")
    start = time.perf_counter()
    errors = {"LEFT": [], "RIGHT": []}
    failure_dumped = {"LEFT": False, "RIGHT": False}
    for frame_index in range(num_frames):
        left_image, right_image = get_frame_pair()
        retries = 0
        while left_image is None and retries < 100:
            time.sleep(0.01)
            left_image, right_image = get_frame_pair()
            retries += 1
        if left_image is None:
            print(f"frame {frame_index:03d}: no camera frame available")
            continue
        timestamp_ms = int((time.perf_counter() - start) * 1000)
        for side, image, mp_landmarker, onnx_tracker in (
            ("LEFT", left_image, mediapipe_left, onnx_left),
            ("RIGHT", right_image, mediapipe_right, onnx_right),
        ):
            mp_points = detect_mediapipe_video(mp_landmarker, image, timestamp_ms)
            onnx_points = detect_onnx(onnx_tracker, image)
            if mp_points is None or onnx_points is None:
                palms = onnx_tracker.detect_palms(image)
                print(
                    f"frame {frame_index:03d} {side}: "
                    f"mp_detected={mp_points is not None} onnx_detected={onnx_points is not None} "
                    f"onnx_palm_candidates={len(palms)}"
                )
                if not failure_dumped[side]:
                    debug_path = output_dir / f"{side.lower()}_tracking_failure_frame{frame_index:03d}.png"
                    cv2.imwrite(str(debug_path), image)
                    print(f"  saved debug frame to {debug_path}")
                    failure_dumped[side] = True
                continue
            diff = np.linalg.norm(mp_points[:, :2] - onnx_points[:, :2], axis=1)
            errors[side].append(diff.mean())
            print(
                f"frame {frame_index:03d} {side}: "
                f"mean_error={diff.mean():6.2f}px max_error={diff.max():6.2f}px"
            )
    for side, side_errors in errors.items():
        if not side_errors:
            print(f"{side}: no frames with detections on both backends.")
            continue
        side_errors = np.asarray(side_errors)
        half = len(side_errors) // 2
        first_half = side_errors[:half].mean() if half else float("nan")
        second_half = side_errors[half:].mean()
        print(
            f"{side} first_half_mean={first_half:.2f}px "
            f"second_half_mean={second_half:.2f}px "
            f"drift(second-first)={second_half - first_half:+.2f}px"
        )


def main():
    args = parse_args()
    for path in (args.model, args.detector, args.landmarks):
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = None
    pipeline = None
    mediapipe_landmarker = None
    try:
        import depthai as dai

        device = dai.Device()
        pipeline, left_queue, right_queue, rectification = create_camera_pipeline(device)
        synchronizer = StereoFrameSynchronizer(left_queue, right_queue)
        pair = None
        while pair is None:
            pair = synchronizer.poll()

        left_frame, right_frame = pair
        left, right = rectify_frames(
            left_frame.getCvFrame(),
            right_frame.getCvFrame(),
            rectification,
        )
        left_small = cv2.resize(
            left,
            (INFERENCE_WIDTH, INFERENCE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        right_small = cv2.resize(
            right,
            (INFERENCE_WIDTH, INFERENCE_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        cv2.imwrite(str(args.output_dir / "left_input.png"), left_small)
        cv2.imwrite(str(args.output_dir / "right_input.png"), right_small)

        mediapipe_landmarker = create_mediapipe_landmarker(args.model)
        onnx_left = OnnxHandTracker(
            args.detector,
            args.landmarks,
            tracking=False,
        )
        onnx_right = OnnxHandTracker(
            args.detector,
            args.landmarks,
            tracking=False,
        )
        compare(
            "LEFT",
            detect_mediapipe(mediapipe_landmarker, left_small),
            detect_onnx(onnx_left, left_small),
        )
        compare_roi_outputs(
            "LEFT",
            detect_mediapipe_on_onnx_roi(
                mediapipe_landmarker,
                left_small,
                onnx_left,
            ),
            detect_onnx(onnx_left, left_small),
        )
        inspect_onnx_pipeline(
            onnx_left,
            left_small,
            "LEFT",
            args.output_dir,
        )
        compare_tflite_onnx_landmark(
            args.model,
            onnx_left,
            left_small,
            "LEFT",
        )
        compare_tflite_onnx_detector(
            args.model,
            onnx_left,
            left_small,
            "LEFT",
        )
        compare(
            "RIGHT",
            detect_mediapipe(mediapipe_landmarker, right_small),
            detect_onnx(onnx_right, right_small),
        )
        compare_roi_outputs(
            "RIGHT",
            detect_mediapipe_on_onnx_roi(
                mediapipe_landmarker,
                right_small,
                onnx_right,
            ),
            detect_onnx(onnx_right, right_small),
        )
        inspect_onnx_pipeline(
            onnx_right,
            right_small,
            "RIGHT",
            args.output_dir,
        )
        compare_tflite_onnx_landmark(
            args.model,
            onnx_right,
            right_small,
            "RIGHT",
        )
        compare_tflite_onnx_detector(
            args.model,
            onnx_right,
            right_small,
            "RIGHT",
        )

        if args.tracking_frames > 0:
            def next_inference_frames():
                pair = synchronizer.poll()
                if pair is None:
                    return None, None
                left_frame, right_frame = pair
                left_full, right_full = rectify_frames(
                    left_frame.getCvFrame(),
                    right_frame.getCvFrame(),
                    rectification,
                )
                left_frame_small = cv2.resize(
                    left_full,
                    (INFERENCE_WIDTH, INFERENCE_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                right_frame_small = cv2.resize(
                    right_full,
                    (INFERENCE_WIDTH, INFERENCE_HEIGHT),
                    interpolation=cv2.INTER_AREA,
                )
                return left_frame_small, right_frame_small

            print("\nKeep your hand in view for the tracking-mode sequence...")
            mediapipe_video_left = create_mediapipe_landmarker(
                args.model, running_mode=vision.RunningMode.VIDEO
            )
            mediapipe_video_right = create_mediapipe_landmarker(
                args.model, running_mode=vision.RunningMode.VIDEO
            )
            onnx_track_left = OnnxHandTracker(args.detector, args.landmarks, tracking=True)
            onnx_track_right = OnnxHandTracker(args.detector, args.landmarks, tracking=True)
            try:
                compare_tracking_sequence(
                    mediapipe_video_left,
                    mediapipe_video_right,
                    onnx_track_left,
                    onnx_track_right,
                    next_inference_frames,
                    args.tracking_frames,
                    args.output_dir,
                )
            finally:
                mediapipe_video_left.close()
                mediapipe_video_right.close()
    finally:
        if mediapipe_landmarker is not None:
            mediapipe_landmarker.close()
        if pipeline is not None:
            pipeline.stop()
        if device is not None:
            device.close()


if __name__ == "__main__":
    main()
