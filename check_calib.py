import depthai as dai
import numpy as np

with dai.Device() as device:
    calib = device.readCalibration()

    print("=== Left CAM_B ===")
    left_k = np.array(
        calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_B,
            1280,
            800,
        )
    )
    print(left_k)

    print("\n=== Right CAM_C ===")
    right_k = np.array(
        calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_C,
            1280,
            800,
        )
    )
    print(right_k)

    print("\n=== Left distortion ===")
    print(
        calib.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_B
        )
    )

    print("\n=== Right distortion ===")
    print(
        calib.getDistortionCoefficients(
            dai.CameraBoardSocket.CAM_C
        )
    )

    print("\n=== Left -> Right extrinsics ===")
    extrinsics = np.array(
        calib.getCameraExtrinsics(
            dai.CameraBoardSocket.CAM_B,
            dai.CameraBoardSocket.CAM_C,
        )
    )
    print(extrinsics)

    print("\n=== Baseline [cm] ===")
    print(
        calib.getBaselineDistance(
            dai.CameraBoardSocket.CAM_B,
            dai.CameraBoardSocket.CAM_C,
        )
    )