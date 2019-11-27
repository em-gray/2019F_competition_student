import roslib
import sys
import rospy
import cv2 as cv
import numpy as np
import time

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
rospy.init_node("toImageMsg")

pub = rospy.Publisher("practice_plates", Image, queue_size=3)

frame = cv.imread("car14.jpg")

frameMsg = bridge.cv2_to_imgmsg(frame)

while (1):
    rospy.sleep(5.)
    try:
        pub.publish(frameMsg)
    except KeyboardInterrupt:
        print("shutting down")

