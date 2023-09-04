import cv2
import torch

def videolocalize(directory,modelPath):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # Load YOLOv5 model
    model = torch.hub.load(directory, 'custom', path=modelPath, source='local')
    flow=0
    while True:
        sucess, frame = cap.read()
        # Detect objects in the frames
        results = model(frame)
        print(flow)
        # Get the bounding box coordinates and labels of detected objects
        bboxes = results.xyxy[0].numpy()
        labels = results.names[0]
        count=0
        # Loop over the detected objects and draw bounding boxesimport
        for i, bbox in enumerate(bboxes):
            # Get the coordinates of the top-left and bottom-right corners of the bounding box
            x1, y1, x2, y2 = bbox[:4].astype(int)

            # Draw the bounding box rectangle and label text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, labels[int(bbox[5])], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.imshow('frame',frame)
            x=(x1+x2)/2
            y=(y1+y2)/2
            point=[]
            point.append(x)
            point.append(y)
            count+=1
            point.insert(0,count)
            print(point)
        #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
        flow+=1
    cap.release()
    cv2.destroyAllWindows()
    #return point
'''
def localize(directory,modelPath,imgPath):
    # Load YOLOv5 model
    model = torch.hub.load(directory, 'custom', path=modelPath, source='local')

    # Load image
    image = cv2.imread(imgPath)

    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects in the image
    results = model(image)

    # Get the bounding box coordinates and labels of detected objects
    bboxes = results.xyxy[0].numpy()
    labels = results.names[0]
    count=0
    # Loop over the detected objects and draw bounding boxesimport
    for i, bbox in enumerate(bboxes):
        # Get the coordinates of the top-left and bottom-right corners of the bounding box
        x1, y1, x2, y2 = bbox[:4].astype(int)

        # Draw the bounding box rectangle and label text
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.putText(image, labels[int(bbox[5])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite('detected_image.jpg',image)
        x=(x1+x2)/2
        y=(y1+y2)/2
        point=[]
        point.append(x)
        point.append(y)
        count+=1
    point.insert(0,count)
    return point
'''
directory='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/'
model='C:/Users/Prajwal/Python_Projects/Masters_Thesis/yolo_deployment/rec_cir-best.pt'
videolocalize(directory,model)