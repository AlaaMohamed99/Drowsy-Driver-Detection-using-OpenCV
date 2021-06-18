from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
from drowsy import *
from  utils import  *
import headpose_utils as headpose
from scipy.spatial import distance
from PIL import Image
import tensorflow as tf
from phone.phone_detection import detect
from tensorflow.python.saved_model import tag_constants
import os
import time


model_path = os.path.join(os.getcwd() +'/phone/checkpoints/','yolov4_tiny_416')


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor.dat')

saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

frame_counter=0
frame_counter_head=0
alarm = False

vs = VideoStream(src=0).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    #function to detect if there's a phone in the screen and raises alarm immediatly
    phone_found = detect(frame,infer)  
    
    if phone_found:
            cv2.putText(frame, "Put Phone Away" , (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            t = Thread(target=sound_alarm, args=['alarm.mp3'])
            t.deamon = True
            t.start()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape_marks = predictor(gray, rect)
        shape = shape_to_np(shape_marks)
        
        
        #detect the angle of the face, and if the driver isn't looking forward for an extended period of time raise an alarm
        angles ,pose_alarm, frame_counter_head = headpose.HeadPoseAngles(shape_marks, frame , frame_counter_head)
        
        
        # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_ascpect_ratio(leftEye)
        rightEAR = eye_ascpect_ratio(rightEye)
        
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # compute the convex hull for eyes, and draw it
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #function to count the frames in which the eye is closed, and rasies alarm if it exceeds the max_no_of_frames
        frame_counter, alarm = drowsy_detect(ear,frame_counter)
        
        #alarm for eyes closed
        if alarm:
            cv2.putText(frame, "WAKE UP!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)            
            t = Thread(target=sound_alarm, args=['alarm.mp3'])
            t.deamon = True
            t.start()
        
        #alarm for not looking forward 
        if pose_alarm:
            cv2.putText(frame, "Please Look  Forward" , (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            t = Thread(target=sound_alarm, args=['alarm.mp3'])
            t.deamon = True
            t.start()
            
        
    cv2.imshow('frame',frame)

    #close the window when q is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
            

