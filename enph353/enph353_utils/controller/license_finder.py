import roslib
import sys
import rospy
import cv2 as cv
import numpy as np
import re

from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from imutils.object_detection import non_max_suppression
import pytesseract
import imutils
import operator
#from keras.models import load_model

#From PyImageSearch
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

# ROSNode to process camera input and publish license plate files, while keeping track of visited plates

# main
class license_finder:
	def __init__(self):
		self.bridge = CvBridge()
		self.license_pub = rospy.Publisher("/license_plate", String)
		self.sub = rospy.Subscriber("license_pics", Image, self.get_plate)
		self.net_path = "frozen_east_text_detection.pb"
		self.padding = 0.05
		self.visited = []

		# "visited plates" duplicate handlers
		self.p1 = {}
		self.p2 = {}
		self.p3 = {}
		self.p4 = {}
		self.p5 = {}
		self.p6 = {}
		self.p0 = {}

		# common letter mistakes
		self.mistakenChars = {
			"O":'0',
			"T":'1',
			"I":'1',
			"S":'5',
			"G":'6',
			"H":'4',
			"Z":'2',
			"@":'0'
			}
	
		# common number mistakes
		self.mistakenNums = {
			"0":'O',
			"1":'T',
			"8":'S',
			"6":'G',
			"4":'H',
			"2":'Z',
			"7":'T',
			"3":'J'
			}

		initial_msg = String()
		initial_msg.data = "Jules&Em,Securus,0,1234"
		self.license_pub.publish(initial_msg)
	
	def plate_handler(self, plate, spot):
		if plate in spot:
			spot[plate] += 1
		else:
			spot[plate] = 1


	# Inspiration from http://projectsfromtech.blogspot.com/2017/10/visual-object-recognition-in-ros-using.html
	def get_plate(self, data):
		frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
		print("got the frame")
		# except CvBridgeError as e:
		#     print(e)

		image = cv.resize(frame, None, fx=1, fy=2, interpolation = cv.INTER_LINEAR)
		(H, W) = image.shape[:2]
		image = image[H/2:4*H/5, 0:W/3]
		#image = image[H/2:4*H/5, 0:W/3]
		#cv.imshow("cropped", image)
		#cv.waitKey(0)
		orig = image.copy()
		(origH, origW) = image.shape[:2]

		# set the new width and height and then determine the ratio in change
		# for both the width and height
		# (newW, newH) = (args["width"], args["height"])
		rW = origW / float(320)
		rH = origH / float(320)

		# resize the image and grab the new image dimensions
		image = cv.resize(image, (320, 320))
		(H, W) = image.shape[:2]

		# define the two output layer names for the EAST detector model that
		# we are interested -- the first is the output probabilities and the
		# second can be used to derive the bounding box coordinates of text
		layerNames = [
			"feature_fusion/Conv_7/Sigmoid",
			"feature_fusion/concat_3"]

		# load the pre-trained EAST text detector
		print("[INFO] loading EAST text detector...")
		net = cv.dnn.readNet(self.net_path)

		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		blob = cv.dnn.blobFromImage(image, 1.0, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		# initialize the list of results
		results = []

		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# in order to obtain a better OCR of the text we can potentially
			# apply a bit of padding surrounding the bounding box -- here we
			# are computing the deltas in both the x and y directions
			dX = int((endX - startX) * self.padding)
			dY = int((endY - startY) * 2*self.padding)

			# apply padding to each side of the bounding box, respectively
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))

			# extract the actual padded ROI
			roi = orig[startY:endY, startX:endX]

			# further processing
			gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
			gray = cv.threshold(gray, 0, 255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
			gray = cv.medianBlur(gray, 3)
			# cv.imshow("mask", ]gray)
			# cv.waitKey(0)


			# in order to apply Tesseract v4 to OCR text we must supply
			# (1) a language, (2) an OEM flag of 4, indicating that the we
			# wish to use the LSTM neural net model for OCR, and finally
			# (3) an OEM value, in this case, 7 which implies that we are
			# treating the ROI as a single line of text
			config = ("-l eng --oem 1 --psm 7")
			text = pytesseract.image_to_string(gray, config=config)

			# add the bounding box coordinates and OCR'd text to the list
			# of results
			results.append(((startX, startY, endX, endY), text))
	 
		# sort the results bounding box coordinates from top to bottom
		results = sorted(results, key=lambda r:r[0][1])
		infoString = ""

		# loop over the results
		for ((startX, startY, endX, endY), text) in results:
			# display the text OCR'd by Tesseract
			#print("OCR TEXT")
			#print("========")
			#print(str(unicode(text).encode('utf-8')))

			# strip out non-ASCII text so we can draw the text on the image
			# using OpenCV, then draw the text and a bounding box surrounding
			# the text region of the input image
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			infoString = infoString + text
			output = orig.copy()

			cv.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
			cv.putText(output, text, (startX, startY - 20),
			cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
		#cv.imshow("output", output)
		#cv.waitKey(0)

		
			# remove all non-alphanumerics
		original = infoString
		print("original: " + infoString)
		infoString = re.sub(r'\W+', '', infoString)
		infoString = infoString.upper()
		infoString = list(infoString)
		spot = None
		message = String()
		try:
			location = infoString[1]
			location_copy = location
			if location == "O":
				spot = 0
			if location in self.mistakenChars:
				location = self.mistakenChars[location]
			if location == "1":
				spot = 1
			elif location == "2":
				spot = 2
			elif location == "3":
				spot = 3
			elif location == '4':
				spot = 4
			elif location == "5":
				spot = 5
			
		except:
			pass

		if len(infoString) == 6:
			# correct issues in position reading
			if location in self.mistakenChars:
				location = self.mistakenChars[location]
			if int(location) == 8 or int(location) == 6:
				location = 5
			# handler for when it reads "PS"
			
			# correct issues in plate reading
			plate = infoString[-4:]
			for char in plate:
				idx = plate.index(char)
				if idx < 2:
					if char in self.mistakenNums:
						char = self.mistakenNums[char]
				else:
					if char in self.mistakenChars:
						char = self.mistakenChars[char]

				# replace originalchar  with corrected
				plate[idx] = char

			# handle different valid plates (this is also going to be cleaned up)
			
			plateStr = ''.join(plate)
			if spot == 1:
				self.plate_handler(plateStr, self.p1)
				bestPlate = max(self.p1.iteritems(), key=operator.itemgetter(1))[0]
					
			elif spot == 2:
				self.plate_handler(plateStr, self.p2)
				bestPlate = max(self.p2.iteritems(), key=operator.itemgetter(1))[0]

			elif spot == 3:
				self.plate_handler(plateStr, self.p3)
				bestPlate = max(self.p3.iteritems(), key=operator.itemgetter(1))[0]

			elif spot == 4:
				self.plate_handler(plateStr, self.p4)
				bestPlate = max(self.p4.iteritems(), key=operator.itemgetter(1))[0]

			elif spot == 5:
				self.plate_handler(plateStr, self.p5)
				bestPlate = max(self.p5.iteritems(), key=operator.itemgetter(1))[0]
		
			elif spot == 0:
				self.plate_handler(plateStr, self.p0)
				bestPlate = max(self.p0.iteritems(), key=operator.itemgetter(1))[0]
			
			# check if bestPlate is valid
			try:
				plateList = list(bestPlate)
				sendMsg = True
				for char in plateList:
					idx = plateList.index(char)
					if idx < 2:
						if str.isdigit(str(char)):
							sendMsg = False
					elif str.isalpha(str(char)):
							sendMsg = False

				if sendMsg:
					message.data = "Jules&Em,Securus," + str(location) + "," + bestPlate
					print("spot: " + str(spot) + "   original :" + str(original) + "   best plate: " + str(bestPlate))
					#print(message.data)
					self.license_pub.publish(message)
			except:
				pass


def control():
	rospy.init_node('license_finder', anonymous=True)
	lf = license_finder()
	

	try:
		rospy.spin()

	except KeyboardInterrupt:
		print("Shutting down")

	cv.destroyAllWindows()

if __name__ == '__main__':
	try:
		control()
	except rospy.ROSInterruptException: pass
