

from simple_pid import PID
import cv2
import torch
import threading
import rtde_control
import rtde_receive
import time

pidx = PID(0.0003, 0.0, 0.0001, setpoint=320)
pidy=PID(-0.0003, 0.0, 0.0001, setpoint=240)
#pid.setpoint = 320
pidx.sample_time = 0.01
pidy.sample_time = 0.01

rtde_r = rtde_receive.RTDEReceiveInterface("169.254.37.99",100.0,verbose=True)
rtde_c = rtde_control.RTDEControlInterface("169.254.37.99")
directory='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/'
weights='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/rec_cir-best.pt'#Directory for yolo object weights
position=[0,0]
capPose=[-0.200, -0.611, 0.150, 0.032, -3.154, -0.017]###########################################
def drive(control):
    rtde_c.speedL(control, 0.5, 0.5)

model = torch.hub.load(f'{directory}', 'custom', path=f'{weights}', source='local')
#write object detection code here and update position in a loop
rtde_c.moveL(capPose,0.5,0.5)
cap=cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

while True:
    ret, frame = cap.read()
    if not ret:
        print('No frame is detected')
        break
    results = model(frame)
    if results == []:
        pass
    else:
        # Get the bounding box coordinates and labels of detected objects
        bboxes = results.xyxy[0].numpy() #if bbox is empty = no detection = pass
        labels = results.names[0]
        for i, bbox in enumerate(bboxes):
            # Get the coordinates of the top-left and bottom-right corners of the bounding box
            x1, y1, x2, y2 = bbox[:4].astype(int)
            confidence = round(float(bbox[4]), 2)
            label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
            object=[x1/2+x2/2,y1/2+y2/2]
            position=object
            # Draw the bounding box rectangle and label text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            vel_x=pidx(position[0])
            vel_y=pidy(position[1])
            #drive([vel_x,0,vel_y,0,0,0])
            drive([vel_x,vel_y,0,0,0,0])
        cv2.imshow('frame',frame)
        if (319.5<=position[0]<=320.5) and (239.5<=position[1]<=240.5):
            camera_pose=rtde_r.getActualTCPPose()
            rtde_c.disconnect()
            rtde_c.reconnect()
            break
        if cv2.waitKey(1) & 0xFF==27:
            break
cap.release()
cv2.destroyAllWindows()
#Write program
rtde_c.disconnect()
rtde_r.disconnect()