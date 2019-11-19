import roslib
import sys
import rospy
import cv2 as cv
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

d = 0.1
speed = 0.01

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=30)
        self.prev_err = 0

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height = np.size(frame, 0)
        width = np.size(frame,1)
        #print(height, width)

        
        roi_right = frame[height-300:height, width-640:width] #region of interest
        gray = cv.cvtColor(roi_right, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

        # #hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # lower_grey = np.array([65,65,65])
        # upper_grey = np.array([140,140,140])
        # mask = cv.inRange(frame, lower_grey, upper_grey)

        M = cv.moments(thresh)

        meep = (M["m00"])

        if meep < 1:
            meep = meep + 1
        cX = int(M["m10"] / meep)+640
        print("cX:", cX)

        drive_error = cX - 960
        print("drive error:", drive_error)
        velocity = Twist()
        if (drive_error > 3) or (drive_error < -3):
            velocity.angular.z = d * drive_error
            print("angular v:", velocity.angular.z)
            self.vel_pub.publish(velocity)
        else:
            velocity.linear.x = speed
            self.vel_pub.publish(velocity)

        #print(velocity.linear.x)
        cv.circle(frame, (int(cX), height-100), 20, (0, 255, 0), -1)
        cv.imshow("centroid view", gray)
        cv.imshow("Robot Camera", frame)
        cv.waitKey(1)

def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    