import depthai as dai

print("DepthAI version:", dai.__version__)

devices = dai.Device.getAllAvailableDevices()

print("Devices:", len(devices))

for device in devices:
    print(device)