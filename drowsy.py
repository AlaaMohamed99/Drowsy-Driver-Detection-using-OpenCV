import cv2

def drowsy_detect(ear,frame_counter):
	threshold = 0.3
	max_no_of_frames = 48
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

	# if ear < threshold:
	# 	frame_counter += 1
	# 	print(frame_counter)
	# 	if frame_counter >= max_no_of_frames:
	# 		if not alarm:
	# 			alarm = True
	# 			t = Thread(target=sound_alarm, args=['alarm.mp3'])
	# 			t.deamon = True
	# 			t.start()
	# 			print(frame_counter)
	# 			#return frame_counter
	# else:
	# 	frame_counter = 0
	# 	alarm = False
	# 	#return frame_counter