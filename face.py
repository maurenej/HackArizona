# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

def check_blink_thresh(inc, tot, elapsed):
	global start_time
	global top
	global INC_TOTAL
	global TOTAL
	global NUM_ITERATIONS
	global TO_TIME
	print("ITERATION")
	bpm = tot/ elapsed
	if tot == 0 or (bpm < BPM_THRESH and ((inc/tot) > 0.5)) or NUM_ITERATIONS == 3:
		print("Desktop is going to sleep")
		time.sleep(2) #delays monitor going off by 2 seconds
		os.system("pmset displaysleepnow")
		TO_TIME = 1
		INC_TOTAL = 0
		TOTAL = 0
		NUM_ITERATIONS = 1
	else:
		NUM_ITERATIONS = NUM_ITERATIONS + 1
		TO_TIME = 1
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, 
	help="path to input video file")
ap.add_argument("--min", type=float, default=20, help="minimum time of inactivity")
args = vars(ap.parse_args())

AVG_EAR = 0
start_time = 0

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold

# also includes threshold for an INCOMPLETE blink
EYE_AR_THRESH = 0.23
EYE_AR_INC_THRESH = 0.1
EYE_AR_CONSEC_FRAMES = 3
BPM_THRESH = 10
MIN_TIME = args["min"]
NUM_ITERATIONS = 1
 
# initialize the frame counters and the total number of blinks
COUNTER = 0
INC_COUNTER = 0
TOTAL = 0
INC_TOTAL = 0
TO_TIME = 1

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)


#start_time = time.time()
CALIBRATING = True
calibrate_count = 1
print("Calibrating...")

while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		jaw = shape[jStart:jEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		jawHull = cv2.convexHull(jaw)
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if CALIBRATING:
			AVG_EAR = ((AVG_EAR * calibrate_count) + ear) / (calibrate_count + 1)
			calibrate_count += 1
			if calibrate_count == 50:
				CALIBRATING = False
				EYE_AR_THRESH = 0.75 * AVG_EAR
		
		else:
			if (TO_TIME):
				start_time = time.time()
				TO_TIME = 0
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			
 
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				
 
				# reset the eye frame counter
				COUNTER = 0
		
			# do the same process to record incomplete blinks
			if ear > EYE_AR_INC_THRESH and ear < EYE_AR_THRESH:
				INC_COUNTER += 1
			
			else:
				if INC_COUNTER >= EYE_AR_CONSEC_FRAMES:
					INC_TOTAL += 1
				
				INC_COUNTER = 0

			# The elapsed time is the current time minus the start time (divided by 60 for minutes)
			elapsed_time = float((time.time() - start_time) / 60.0)	

			if elapsed_time > MIN_TIME:
				#sleep(5)
				check_blink_thresh(INC_TOTAL, TOTAL, elapsed_time)
				#start_time = time.time()


			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			# cv2.putText(frame, "Blinks: {}".format(INC_TOTAL), (10, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			# cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



