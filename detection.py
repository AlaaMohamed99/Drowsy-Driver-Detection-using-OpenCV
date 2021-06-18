from scipy.spatial import distance
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

#cap = cv2.VideoCapture(0)


ALARM_ON = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def eye_ascpect_ratio(points):
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])

    EAR = (A+B)/(2*C)

    return EAR
def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)


# def detect():
# 	detector = dlib.get_frontal_face_detector()

# 	return detector
# def predict():
# 		predictor = dlib.shape_predictor('shape_predictor.dat')
# 		return predictor



def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)


def drowsy_detect(ear,frame,frame_counter):
	threshold = 0.3
	#frame_counter=0
	max_no_of_frames = 48
	ALARM_ON= False
	if ear < threshold:
		#global frame_counter
		frame_counter += 1
		if frame_counter >= max_no_of_frames:
			
			if not ALARM_ON:
				alarm = True
				t =Thread(target=sound_alarm('Alarm.mp3'),args="Alarm.mp3")
				t.deamon = True
				t.start()
				cv2.putText(frame, "TAKE CARE!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	else:
		frame_counter = 0
		alarm = False

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-a", "--alarm", type=str, default="",
# 	help="path alarm .WAV file")
# ap.add_argument("-w", "--webcam", type=int, default=0,
# 	help="index of webcam on system")
# args = vars(ap.parse_args())

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
###############################


vs = VideoStream(src=0).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	detector = dlib.get_frontal_face_detector()
	rects = detector(gray, 0)
	faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
      
    )
    # draw rectangle around the face
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		predictor = dlib.shape_predictor('shape_predictor.dat')
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
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

        #Alarm#
		#qqqqqdrowsy_detect(ear,frame,0)
		# if ear < EYE_AR_THRESH:
		# 	frame_counter += 1
		# 	if frame_counter >= EYE_AR_CONSEC_FRAMES:
		# 		if not ALARM_ON:
		# 			ALARM_ON = True
		# 			# check to see if an alarm file was supplied,
		# 			# and if so, start a thread to have the alarm
		# 			# sound played in the background
		# 			if not ALARM_ON:
		# 				t = Thread(target=sound_alarm,args=["Alarm.mp3"])
		# 				t.deamon = True
		# 				t.start()
		# 		# draw an alarm on the frame
		# 		cv2.putText(frame, "Take Care!", (10, 30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		# else:
		# 	frame_counter = 0
		# 	ALARM_ON = False

	
	cv2.imshow('frame',frame)

     
	key = cv2.waitKey(1) & 0xFF
#To quit and exit
	if key == ord("q"):
		break
# do a bit of cleanup



cv2.destroyAllWindows()
vs.stop()
	