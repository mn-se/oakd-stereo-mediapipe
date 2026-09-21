import cv2
import depthai as dai


pipeline = dai.Pipeline()

# OAK-D:
# CAM_B = Left mono
# CAM_C = Right mono
left = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_B
)

right = pipeline.create(dai.node.Camera).build(
    dai.CameraBoardSocket.CAM_C
)

# Request grayscale output.
left_queue = left.requestOutput(
    (1280, 800),
    dai.ImgFrame.Type.GRAY8,
).createOutputQueue(
    maxSize=4,
    blocking=False,
)

right_queue = right.requestOutput(
    (1280, 800),
    dai.ImgFrame.Type.GRAY8,
).createOutputQueue(
    maxSize=4,
    blocking=False,
)

# Start pipeline.
pipeline.start()

while pipeline.isRunning():

    left_frame = left_queue.tryGet()
    right_frame = right_queue.tryGet()

    if left_frame is not None:
        frame_left = left_frame.getCvFrame()
        cv2.imshow("Left", frame_left)

    if right_frame is not None:
        frame_right = right_frame.getCvFrame()
        cv2.imshow("Right", frame_right)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()