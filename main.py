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
from drowsy import drowsy_detect
from  utils import  *
import headpose_utils as headpose




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor.dat')
    
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

threshold = 0.3
max_no_of_frames = 48

frame_counter=0
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
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        #reeeem ............................................................................................
        angles ,pose_alarm,GAZE = headpose.HeadPoseAngles(shape, frame)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_ascpect_ratio(leftEye)
        rightEAR = eye_ascpect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        frame_counter, alarm = drowsy_detect(ear,frame_counter)

        if alarm:
            t = Thread(target=sound_alarm, args=['alarm.mp3'])
            t.deamon = True
            t.start()
            cv2.putText(frame, "WAKE UP!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if pose_alarm:
            
            t = Thread(target=sound_alarm, args=['alarm.mp3'])
            t.deamon = True
            t.start()
            cv2.putText(frame, GAZE , (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        
            
        # `cv2.putText(image, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        # cv2.imshow("Head Pose", image)    
        
            
    cv2.imshow('frame',frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
            

