#The direction for the movements should be changed according to the setup
import cv2
import matplotlib.pyplot as plt
import torch
import threading
import rtde_control
import rtde_receive
import time
import numpy as np
#directory='C:/Users/sarojd/Vision_Arsenal/bisection_NEW/'#Directory for yolo deployment files
#weights='C:/Users/sarojd/Vision_Arsenal/bisection_NEW/pendrive.pt'#Directory for yolo object weights
object_coordinates = []
timestamps = []
position=[0,0]
goalX=0
goalY=0
directory='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/'
weights='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/rec_cir-best.pt'
#weights='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/ecc.pt'

def start(capPose):
    rtde_r1 = rtde_receive.RTDEReceiveInterface("169.254.37.99",100.0,verbose=True)
    rtde_c1 = rtde_control.RTDEControlInterface("169.254.37.99")
    rtde_c1.moveL(capPose, 0.15, 0.1)
    
    
    def video(directory,weights):
        global position
        model = torch.hub.load(f'{directory}', 'custom', path=f'{weights}', source='local')
        #write object detection code here and update position in a loop
        cap=cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print('No frame is detected')
                break
            results = model(frame)
            # Get the bounding box coordinates and labels of detected objects
            bboxes = results.xyxy[0].numpy()
            labels = results.names[0]
            for i, bbox in enumerate(bboxes):
                # Get the coordinates of the top-left and bottom-right corners of the bounding box
                x1, y1, x2, y2 = bbox[:4].astype(int)
                confidence = round(float(bbox[4]), 2)
                label = f"{labels[int(bbox[5])]}: {confidence}:[{x1/2+x2/2},{y1/2+y2/2}]"
                object=[x1/2+x2/2,y1/2+y2/2]
                object_coordinates.append(object)
                timestamps.append(time.time())
                # Draw the bounding box rectangle and label text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('frame',frame)
            if object:
                position=object
            if cv2.waitKey(1) & 0xFF==27:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def get_pix():
        global position
        return position
    
    def xp(px,k):
        #this runs if there is positive error
        if px>320:
            #Move(+k)
            pos1=rtde_r1.getActualTCPPose()
            pos1[0]-=k
            rtde_c1.moveL(pos1, 0.15, 0.1)
            #Call@1
            pix= get_pix()
            if pix[0]<320:
                k=k/2
                xn(pix[0],k)
            elif pix[0]>320:
                xp(pix[0],k)
            else :
                return 1
    
    def xn(px,k):
        #this runs if there is negative error
        if px<320:
            #Move(-k)
            pos=rtde_r1.getActualTCPPose()
            pos[0]+=k
            rtde_c1.moveL(pos, 0.15, 0.1)
            #Call@1
            pix= get_pix()
            if pix[0]>320:
                k1=k/2
                xp(pix[0],k1)
            elif pix[0]<320:
                xn(pix[0],k)
            else :
                return 1
    
    def yp(py,k):
        #this runs with a positive error
        if py>240:
            #Move(+k)
            pos1=rtde_r1.getActualTCPPose()
            pos1[1]+=k
            rtde_c1.moveL(pos1, 0.15, 0.1)
            #Call@1
            pix= get_pix()
            if pix[1]<240:
                k1=k/2
                yn(pix[1],k1)
            elif pix[1]>240:
                yp(pix[1],k)
            else :
                return 1
    
    def yn(py,k):
        #this runs if there is negative error
        if py<240:
            #Move(-k)
            pos=rtde_r1.getActualTCPPose()
            pos[1]-=k
            rtde_c1.moveL(pos, 0.15, 0.1)
            #Call@1
            pix= get_pix()
            if pix[1]>240:
                k1=k/2
                yp(pix[1],k1)
            elif pix[1]<240:
                yn(pix[1],k)
            else :
                return 1
    
    detection_thread=threading.Thread(target=video,args=(directory,weights))
    detection_thread.daemon=True
    detection_thread.start()
    
    k=0.005 #I can change this valuey
    
    pix= get_pix()
    if pix[0]>315:
        goalX=xp(pix[0],k)
    elif pix[0]<325:
        goalX=xn(pix[0],k)
    else: 
        goalX=1
    if goalX==1:
        print('Done in Y')
    else: print('Done in X with optimism')
    pix= get_pix()
    if pix[1]>235:
        goalY=yp(pix[1],k)
    elif pix[1]<245:
        goalY=yn(pix[1],k)
    else: 
        goalY=1
    if goalY==1:
        print('Done in Y')
    else: print('Done in Y with optimismY')
    
    
    # Extract x and y coordinates for plotting
    x_coordinates = [coord[0] for coord in object_coordinates]
    y_coordinates = [coord[1] for coord in object_coordinates]
    
    rtde_c1.disconnect()
    rtde_r1.disconnect()
