# This code implements YOLOv3 technique in order to detect objects on the video frames.

# import the necessary packages
import numpy as np
import imutils
import time
import cv2
import os


###-----------------------------------------------------### For tracking
def calc_dist(coord_new_frame, coord_prev_frame):
	coord_subtract = np.subtract(coord_new_frame, coord_prev_frame)
	dist = np.sqrt(coord_subtract[0]**2+coord_subtract[1]**2)
	return int(dist)

def draw_box_rectangle(frame, box, color, id):
	# extract the bounding box coordinates
	# box = (x, y, w, h, centerx, centery)
	(x, y) = (box[0], box[1])
	(w, h) = (box[2], box[3])

	# draw a bounding box rectangle and label on the frame
	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	text = "{}: {:d}".format('Human #',
		int(id))
	cv2.circle(frame, (box[4], box[5]), 10, color,2)
	cv2.putText(frame, text, (x, y - 5),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return frame

track_dictionary = {}

def track_human(boxes, track_dictionary):
	# box = (x, y, w, h, centerx, centery, color)
	idx_in_this_frame = []
	if track_dictionary == {}:
		for box in boxes:
			track_dictionary[i] = box
			idx_in_this_frame.append(boxes.index(box))	
	else:
		for box in boxes:
			min_dist = 5000
			for key, value in track_dictionary.copy().items():
				dist = calc_dist(box[4:6], track_dictionary[key][4:6])
				if dist <= min_dist:
					min_dist = dist
					idx = key

			if min_dist <= min(track_dictionary[idx][2:4]):
				track_dictionary[idx][0:6] = box[0:6]	 
				boxes.remove(box)
				idx_in_this_frame.append(idx)
			else:
				# new human
				k = len(track_dictionary)
				idx_in_this_frame.append(k)
				track_dictionary[k] = box
				boxes.remove(box)		
	return track_dictionary, idx_in_this_frame					


###-----------------------------------------------------###

# construct the argument parse and parse the arguments

path_of_file = os.path.abspath(__file__)
os.chdir(os.path.dirname(path_of_file))

inp_video = 'videos/pedestrians.mp4'
thr_param = 0.3
conf_param = 0.5
out_video = 'output/pedestrians_out.mp4'

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(['yolo-coco/weights', "yolov3.weights"])
configPath = os.path.sep.join(['yolo-coco/cfg', "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(inp_video)
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
frame_count = 0
while True:
	frame_count += 1
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (384, 384),  # (192, 192) \ (256, 256) \ (384, 384)
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	centers = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > conf_param and (classID == 0):
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height), centerX, centerY])

				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_param,
		thr_param)

	# ensure at least one detection exists
	if len(idxs) > 0:
		boxes_tracked = []
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			r = np.random.choice(255)
			g = np.random.choice(255)
			b = np.random.choice(255)
			color = (r,g,b)
			boxes[i].append(color)
			boxes_tracked.append(boxes[i])

		track_dictionary, idx_in_this_frame = track_human(boxes_tracked, track_dictionary)

		for key in idx_in_this_frame:
			# print(track_dictionary[key])
			# print(key)
			# print(track_dictionary.keys())
			frame = draw_box_rectangle(frame, track_dictionary[key][0:6], track_dictionary[key][6], key)


	frame = cv2.resize(frame,(W,H))
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(out_video, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))
	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
