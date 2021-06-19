import cv2
from scipy.spatial import distance
def drowsy_detect(ear,frame_counter):
	threshold = 0.3
	max_no_of_frames = 45
	alarm = False
	if ear < threshold:
		frame_counter += 1
		if frame_counter >= max_no_of_frames:
			if not alarm:
				alarm = True
		return frame_counter, alarm
    
	else:
		frame_counter = 0
		alarm = False
		return frame_counter,alarm

def eye_ascpect_ratio(points):
    A = distance.euclidean(points[1], points[5])
    B = distance.euclidean(points[2], points[4])
    C = distance.euclidean(points[0], points[3])

    EAR = (A+B)/(2*C)

    return EAR
