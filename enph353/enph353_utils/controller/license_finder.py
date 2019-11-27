import roslib
import sys
import rospy
import cv2 as cv
import numpy as np

from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
from keras.models import load_model


# ROSNode to process camera input and publish license plate files, while keeping track of visited plates

# main 
class license_finder:
    def __init__(self):
        self.bridge = CvBridge()
        self.license_pub = rospy.Publisher("/license_plate", String)
        self.repeat_flag_pub = rospy.Publisher("/repeat_flag", Bool)
        self.visited = []


    # Inspiration from http://projectsfromtech.blogspot.com/2017/10/visual-object-recognition-in-ros-using.html
    def get_plate(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        image = cv2.resize(frame, None, fx=1, fy=2, interpolation = cv2.INTER_LINEAR)
        (H, W) = image.shape[:2]
        image = image[H/2:4*H/5, 0:W/3]
        #image = image[H/2:4*H/5, 0:W/3]
        # cv2.imshow("cropped", image)
        # cv2.waitKey(0)
        orig = image.copy()
        (origH, origW) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (args["width"], args["height"])
        rW = origW / float(newW)
        rH = origH / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(args["east"])

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
            output = orig.copy()
            cv2.rectangle(output, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(output, text, (startX, startY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # Publish info (needs formatting)
            textMsg = String()
            textMsg.data = str(text)
            self.license_pub.publish(textMsg)

            # show the output image
            #cv2.imshow("Text Detection", cv2.resize(output,None,None,0.5, 0.5))
            #cv2.waitKey(0)

def control():
    lf = license_finder()
    rospy.init_node('license_finder', anonymous=True)
    
    rospy.Subscriber("license_pics", Image, license_finder.get_plate)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")

    cv.destroyAllWindows()

if __name__ == '__main__':
    try:
    control()
    except rospy.ROSInterruptException: pass
    
