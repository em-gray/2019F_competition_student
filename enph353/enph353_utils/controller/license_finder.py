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

# helper functions - maybe move to another file at some point

# edge detection & helpers adapted from https://github.com/Breta01/handwriting-ocr/blob/master/notebooks/page_detection.ipynb
def edges_det(img, min_val, max_val):
    """ Preprocessing (gray, thresh, filter, border) + Canny edge detection """
    img = cv.cvtColor(resize(img), cv.COLOR_BGR2GRAY)

    # Applying blur and threshold
    img = cv.bilateralFilter(img, 9, 75, 75)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 4)

    # Median blur replace center pixel by median of pixels under kelner
    # => removes thin details
    img = cv.medianBlur(img, 11)

    # Add black border - detection of border touching pages
    # Contour can't touch side of image
    img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[0, 0, 0])
    implt(img, 'gray', 'Median Blur + Border')

    return cv.Canny(img, min_val, max_val)

def resize(img, height=800):
    """ Resize image to given height """	    
    rat = height / img.shape[0]	    
    return cv.resize(img, (int(rat * img.shape[1]), height))

def four_corners_sort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right"""
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])

def contour_offset(cnt, offset):
    """ Offset contour because of 5px border """
    cnt += offset
    cnt[cnt < 0] = 0
    return cnt

def find_page_contours(edges, img):
    """ Finding corner points of page contour """
    # Getting contours  
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding biggest rectangle otherwise return original corners
    height = edges.shape[0]
    width = edges.shape[1]
    MIN_COUNTOUR_AREA = height * width * 0.5
    MAX_COUNTOUR_AREA = (width - 10) * (height - 10)

    max_area = MIN_COUNTOUR_AREA
    page_contour = np.array([[0, 0],
                            [0, height-5],
                            [width-5, height-5],
                            [width-5, 0]])

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # Page has 4 corners and it is convex
        if (len(approx) == 4 and
                cv2.isContourConvex(approx) and
                max_area < cv2.contourArea(approx) < MAX_COUNTOUR_AREA):
            
            max_area = cv2.contourArea(approx)
            page_contour = approx[:, 0]

    # Sort corners and offset them
    page_contour = four_corners_sort(page_contour)
    return contour_offset(page_contour, (-5, -5))
# main 
class license_finder:
    def __init__(self):
        self.bridge = CvBridge()
        self.license_pub = rospy.Publisher("/license_plate", String)
        self.repeat_flag_pub = rospy.Publisher("/repeat_flag", Bool)
        self.visited = []
        self.model = load_model('cleverPlate.h5')


    # Inspiration from http://projectsfromtech.blogspot.com/2017/10/visual-object-recognition-in-ros-using.html
    def get_plate(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height = np.size(frame, 0)
        width = np.size(frame,1)

        # edge detection
        edges_image = edges_det(frame, 200, 250)

        # Close gaps between edges (double page clouse => rectangle kernel)
        edges_image = cv.morphologyEx(edges_image, cv2.MORPH_CLOSE, np.ones((5, 11)))
        page_contour = find_page_contours(edges_image, resize(frame))
        cv.imshow(cv.drawContours(resize(frame), [page_contour], -1, (0, 255, 0), 3))

        cv.waitKey(1)

def control():
    lf = license_finder()
    rospy.init_node('license_finder', anonymous=True)
    
    rospy.Subscriber("R1/pi_camera/image_raw", Image, license_finder.get_plate)

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")

    cv.destroyAllWindows()

if __name__ == '__main__':
    try:
    control()
    except rospy.ROSInterruptException: pass
    
