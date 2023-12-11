from simple_pid import PID
import cv2
import torch
import threading
import matplotlib.pyplot as plt
import rtde_control
import rtde_receive
import numpy as np
from realsense_depth import DepthCamera
from robotiq_gripper_control import RobotiqGripper
import time

object_coordinates = []
timestamps = []
#z_coordinates=[]

# Initialize PID controllers
pidx = PID(0.0003, 0.0, 0.0001, setpoint=320)
pidy = PID(-0.0003, 0.0, 0.0001, setpoint=240)
pidz = PID(0.2, 0.0, 0.0001, setpoint=0.150)

pidx.sample_time = 0.01
pidy.sample_time = 0.01
pidz.sample_time = 0.01

# Initialize RTDE Control and Receive interfaces
rtde_r = rtde_receive.RTDEReceiveInterface("169.254.37.99", 100.0, verbose=True)
rtde_c = rtde_control.RTDEControlInterface("169.254.37.99")

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Initialize Gripper 
gripper = RobotiqGripper(rtde_c)

# Initialize the position for X, Y, Z co-ordinates
position = [0, 0, 0]
capPose = [-0.115, -0.525, 0.380, 0.032, -3.154, -0.017]
rtde_c.moveL(capPose, 0.15, 0.1)

# Gripper Activate
gripper.activate()
gripper.open()
#gripper.move(20)

# Load YOLO model
directory = 'C:/Users/Prajwal/Python_Projects/Masters_Thesis/YOLO-V5/'
weights = 'C:/Users/Prajwal/Python_Projects/Masters_Thesis/YOLO-V5/rec_cir-best.pt'
model = torch.hub.load(f'{directory}', 'custom', path=f'{weights}', source='local')

def get_depth_with_surrounding_pixels(depth_frame, x, y, radius=10):
    height, width = depth_frame.shape
    depth = depth_frame[y, x]
    if depth == 0:
        surrounding_pixels = depth_frame[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
        surrounding_pixels = surrounding_pixels[surrounding_pixels != 0]
        if len(surrounding_pixels) > 0:
            depth = np.mean(surrounding_pixels)
        else:
            depth = -1
    return depth

def drive(control):
    rtde_c.speedL(control, 0.5, 0.5)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frame is detected. Camera Index:', cap.get(cv2.CAP_PROP_POS_MSEC))
        break

    ret, depth_frame, color_frame = dc.get_frame()
    if not ret:
        print('No depth frame is detected')
        break

    results = model(frame)
    if results == []:
        continue

    bboxes = results.xyxy[0].numpy()
    labels = results.names[0]

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox[:4].astype(int)
        confidence = round(float(bbox[4]), 2)
        label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
        object = [x1/2+x2/2, y1/2+y2/2, 0]
        position = object

        depth = get_depth_with_surrounding_pixels(depth_frame, int(position[0]), int(position[1]))
        print("Depth:", depth)
        position[2] = depth/1000

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        object_coordinates.append(object)
        #z_coordinates.append(position[2])
        timestamps.append(time.time())
        vel_x = pidx(position[0])
        vel_y = pidy(position[1])
        vel_z = pidz(position[2])
        drive([vel_x, vel_y, vel_z, 0, 0, 0])

    cv2.imshow('frame', frame)

    if (319 <= position[0] <= 321) and (239.5 <= position[1] <= 240.5) or (0.145 <= round(position[2], 3) <= 0.155):
        
        rtde_c.disconnect()
        rtde_r.disconnect()
        rtde_c.reconnect()
        rtde_r.reconnect()
        camera_pose = rtde_r.getActualTCPPose()
        print("Camera Pose:", camera_pose)
        camera_pose[2] = 0.15
        rtde_c.moveL(camera_pose,0.5,0.5)
        pick_loc = rtde_r.getActualTCPPose()
        pick_loc[0] += 0.0365
        pick_loc[1] -= 0.05085
        rtde_c.moveL(pick_loc,0.5,0.5)
        pick_loc[2] = 0.050
        rtde_c.moveL(pick_loc,0.5,0.5)
        print(pick_loc)
        gripper.close()
        pick_z = rtde_r.getActualTCPPose()
        pick_z[2] = 0.20
        rtde_c.moveL(pick_z,0.5,0.5)
        # Location 1
        rtde_c.moveL([0.26405, -0.52588, 0.2, 0.032, -3.154, -0.017],0.15,0.15)
        #rtde_c.moveL([0.26405, -0.52588, 0.025, 0.032, -3.154, -0.017],0.15,0.15)
        # Location 2
        rtde_c.moveL([0.3326, -0.5229, 0.2, 0.032, -3.154, -0.017],0.15,0.15)
        rtde_c.moveL([0.3362, -0.525, 0.0625, 0.032, -3.154, -0.017],0.15,0.15)
        gripper.open()
        rtde_c.moveL([0.3326, -0.5229, 0.2, 0.032, -3.154, -0.017],0.15,0.15)
        rtde_c.disconnect()
        rtde_r.disconnect()
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Extract x and y coordinates for plotting
x_coordinates = [coord[0] for coord in object_coordinates]
y_coordinates = [coord[1] for coord in object_coordinates]
print(x_coordinates)
print(y_coordinates)

#timestamps=np.array(timestamps)
# Plotting
plt.figure()
plt.plot(timestamps, x_coordinates, label='X Coordinate')
plt.plot(timestamps, y_coordinates, label='Y Coordinate')
#plt.plot(timestamps, z_coordinates, label='Z Coordinate')
plt.xlabel('Time')
plt.ylabel('Coordinate Value')
plt.legend()
plt.savefig('plot.png')
plt.show(block=True)

cap.release()
cv2.destroyAllWindows()
rtde_c.disconnect()
rtde_r.disconnect()