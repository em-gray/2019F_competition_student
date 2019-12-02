#!/usr/bin/env python
# Script to identify text in images from ENPH 353 robot competition course, using Google Tesseract and 
# EAST text detection. Adapted from PyImageSearch:
# https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import imutils
import os
import re
import operator



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
			if scoresData[x] < args["min_confidence"]:
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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

for filename in os.listdir(args["image"]):
# load the input image and grab the image dimensions
	image = cv2.imread(args["image"]+"/"+filename)
	image = cv2.resize(image, None, fx=1, fy=2, interpolation = cv2.INTER_LINEAR)
	(H, W) = image.shape[:2]
	image = image[H/2:4*H/5, 0:W/3]
	#image = image[H/2:4*H/5, 0:W/3]
	# cv2.imshow("cropped", image)
	# cv2.waitKey(0)
	orig = image.copy()
	(origH, origW) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height
	# (newW, newH) = (args["width"], args["height"])
	rW = origW / float(320)
	rH = origH / float(320)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (320, 320))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet("frozen_east_text_detection.pb")

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
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
		dX = int((endX - startX) * args["padding"])
		dY = int((endY - startY) * 2*args["padding"])

		# apply padding to each side of the bounding box, respectively
		startX = max(0, startX - dX)
		startY = max(0, startY - dY)
		endX = min(origW, endX + (dX * 2))
		endY = min(origH, endY + (dY * 2))

		# extract the actual padded ROI
		roi = orig[startY:endY, startX:endX]

		# further processing 
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		gray = cv2.medianBlur(gray, 3)
		# cv2.imshow("mask", ]gray)
		# cv2.waitKey(0)


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
		print("OCR TEXT")
		print("========")
		print(str(unicode(text).encode('utf-8')))

		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw the text and a bounding box surrounding
		# the text region of the input image
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		infoString = infoString + text
		output = orig.copy()
		cv2.rectangle(output, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(output, text, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

		# show the output image
		cv2.imshow("Text Detection", cv2.resize(output,None,None,0.5, 0.5))
		cv2.waitKey(0)
	
	# remove all non-alphanumerics
	infoString = re.sub(r'\W+', '', infoString)
	infoString = infoString.upper()
	infoString = list(infoString)
	
	message = ""


	# visited dictionaries (hacky, sorry!)
	p1 = {}
	p2 = {}
	p3 = {}
	p4 = {}
	p5 = {}
	p6 = {}
	p0 = {}

	# common letter mistakes
	mistakenChars = {
			"O":'0',
			"T":'1',
			"S":'8',
			"G":'6',
			"H":'4',
			"Z":'2'	}
	
	# common number mistakes
	mistakenNums = {
			"0":'O',
			"1":'T',
			"8":'S',
			"6":'G',
			"4":'H',
			"2":'Z'
	}

	if len(infoString) == 6:
		# correct issues in location reading
		location = infoString[1]
		if location in mistakenChars:
			location = mistakenChars[location]
			#handler for when it reads "PS"
			if location == 8:
				location = 5

		# correct issues in plate reading
		plate = infoString[-4:]
		for char in plate:
			idx = plate.index(char)
			if idx < 2:
				if char in mistakenNums:
					char = mistakenNums[char]
			else:
				if char in mistakenChars:
					char = mistakenChars[char]

			# replace originalchar  with corrected
			plate[idx] = char

		# handle different valid plates (this is also going to be cleaned up)
		spot = int(location)
		plateStr = ''.join(plate)
		if spot == 1:
			if plateStr in p1:
				p1[plateStr] += 1
			else:
				p1[plateStr] = 1
			print(p1)
			bestPlate = max(p1.iteritems(), key=operator.itemgetter(1))[0]
				
		elif spot == 2:
			if plateStr in p2:
				p2[plateStr] += 1
			else:
				p2[plateStr] = 1
			print(p2)
			bestPlate = max(p2.iteritems(), key=operator.itemgetter(1))[0]
		elif spot == 3:
			if plateStr in p3:
				p3[plateStr] += 1
			else:
				p3[plateStr] = 1
			print(p3)
			bestPlate = max(p3.iteritems(), key=operator.itemgetter(1))[0]
		elif spot == 4:
			if plateStr in p4:
				p4[plateStr] += 1
			else:
				p4[plateStr] = 1
			print(p4)
			bestPlate = max(p4.iteritems(), key=operator.itemgetter(1))[0]
		elif spot == 5:
			if plateStr in p5:
				p5[plateStr] += 1
			else:
				p5[plateStr] = 1
			print(p5)
			bestPlate = max(p5.iteritems(), key=operator.itemgetter(1))[0]
		elif spot == 6:
			if plateStr in p6:
				p6[plateStr] += 1
			else:
				p6[plateStr] = 1
			print(p6)
			bestPlate = max(p6.iteritems(), key=operator.itemgetter(1))[0]
		elif spot == 0:
			if plateStr in p0:
				p0[plateStr] += 1
			else:
				p0[plateStr] = 1
			print(p0)
			bestPlate = max(p0.iteritems(), key=operator.itemgetter(1))[0]
		

		message = "Jules&Em,Securus," + location + "," + bestPlate
	


		print(message)