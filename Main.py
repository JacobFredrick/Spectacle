import os
import sys
import cv2
import subprocess
import numpy as np
import torch

from ultralytics import YOLO


model = YOLO("yolo26n.pt")


init_path = ("init.cmd")
media_path = ("Media/Vid3.mp4")
vid_capture = cv2.VideoCapture(media_path)
ret,frame = vid_capture.read()
WIN_NAME = "Spectacle"
FPS = vid_capture.get(cv2.CAP_PROP_FPS)
tuner_Constants = {"vidS" : 1}
FRAME_DELAY = int(FPS)
ROI = None

def find_matching_detection(results, initial_bbox, iou_threshold=0.5):
    x, y, w, h = initial_bbox
    initial_roi = [x, y, x + w, y + h]
    best_match = None
    max_iou = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            # Calculate Intersection over Union (IoU) (You'd need a helper function for this)
            # A simple way for this example is checking overlap with a custom function or dedicated library
            
            # Placeholder for actual IoU calculation
            # Iou calculation is complex to code from scratch, for this example we'll assume a match based on proximity/overlap
            # In a real application, you might use a dedicated tracking library that handles initial state feeding
            pass # Skipping detailed IOU for brevity, focus on the logic

    # A more practical approach is using the built-in tracking IDs and filtering based on the initial detection
    # The built-in tracker handles association
    return initial_roi # Return the ROI for display in this simplified example

def initWindow():

    cv2.namedWindow(WIN_NAME,cv2.WINDOW_NORMAL)
    cv2.imshow(WIN_NAME,frame)
    cv2.createTrackbar("FPS",WIN_NAME,int(FPS),int(FPS),lambda x: None)
    
def event(event,x,y,flags,param):

    if event == cv2.EVENT_MOUSEMOVE:
        print("move")

def detectKey():
    key = cv2.waitKey(FRAME_DELAY) & 0xFF
    if key == ord('c'):
        return True
    elif key == ord('q'):
        print("testpress")
    elif key == ord('r'):
        select_object_roi(frame)

def clamp(val,minn,maxn):
    return min(max(val,minn),maxn)

def update():
    global FRAME_DELAY
    tuner_Constants["vidS"] = clamp(cv2.getTrackbarPos("FPS",WIN_NAME),1,int(FPS))

    FRAME_DELAY = int(FPS/tuner_Constants["vidS"])

def select_object_roi(frame):
    global ROI
    ROI = cv2.selectROI(WIN_NAME, frame, fromCenter=False, showCrosshair=True)

def run():
    while True:
        global ret,frame
        ret,frame = vid_capture.read()
        if not ret: 
            break
        model = YOLO("yolo26n.pt")
        #results = model(frame)
        #annotate = results[0].plot()
        if not ROI:
            pass
        results = model.track(frame,conf=0.001,persist=True,classes=0)
        for result in results:
            boxes = result.boxes
            if boxes.id is not None:
                for box in boxes:
                    # You'd filter by the specific ID of your selected object here
                    # Example: if box.id == your_target_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID: {int(box.id[0])}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        
        cv2.imshow(WIN_NAME,frame)
        update()
        if detectKey():
            break
        vid_capture.release
    vid_capture.release
    cv2.destroyAllWindows
# Immediately invoked functions --------------------------------------------------

initWindow()
cv2.setMouseCallback(WIN_NAME,event)
run()












# Object tracking vs object detection
# use object detection
# issues:
# Tracking boxes don't delete after required time
# Tracking lines aren't relative to video, are relative to program runtime

# def init():
#     try:
#         result = subprocess.run([init_path], check=True, shell=True, capture_output=True, text=True)
#         print("STDOUT:", result.stdout)
#         print("STDERR:", result.stderr)
#         print(f"Command executed successfully with return code: {result.returncode}")
#         import cv2
        
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with return code: {e.returncode}")
#         print("STDOUT:", e.stdout)
#         print("STDERR:", e.stderr)
#     except FileNotFoundError:
#         print(f"The file {init_path} was not found.")




# init()
