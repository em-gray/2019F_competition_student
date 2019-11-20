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

d = 0.05
speed = 0.01

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=30)
        self.prev_err = 0
        self.crosswalk = False
        self.red = 0       # 0 means i havent seen cross walk, 1 means i saw the first red line, 2 means i am between lines, 3 is i saw second line, 4 is im passed it

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

        roi_crosswalk = frame[height-200:height] # look at 50 rows of frame towards the bottom of screen

        lower_red = np.array([0,0,245])
        upper_red = np.array([10,10,255])
    
        mask = cv.inRange(roi_crosswalk, lower_red, upper_red)
        cv.imshow("Crosswalk detection", mask)

        M = cv.moments(thresh)
        meep = (M["m00"])
        if meep < 1:
            meep = meep + 1
        cX = int(M["m10"] / meep)+640
        
        cv.circle(frame, (int(cX), height-100), 20, (0, 255, 0), -1)
        cv.imshow("Cropped right corner", gray)
        cv.imshow("Robot Camera", frame)
        cv.waitKey(1)

        drive_error = cX - 960
        #print("drive error:", drive_error)
        velocity = Twist()

        if self.red == 0:
            if (drive_error > 3) or (drive_error < -3):
                velocity.angular.z = d * drive_error
                #print("angular v:", velocity.angular.z)
                self.vel_pub.publish(velocity)
            else:
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)


        #self.red == 0 havent seen anything
        # look for red
        # when u see red, set self.red to 1 (saw the red line transition to next step)
        # pause for 5 seconds
            sum = 0
            for i in range (400,800):
                sum += mask[20][i] #numpy arrays are y,x

             # SEE RED ONCE and STOP
            if sum > 0:
                print("SEEING RED 1:", sum)
                self.red = 1
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = 0
                self.vel_pub.publish(velocity)
                time.sleep(5)

        # self.red == 1 saw the first red line
        # need to wait for that region to be black again
        # drive forward untili sum =0 
        # if sum = 0, set self.red to 2 (saw black transition to next)
        elif self.red == 1:
            sum = 0
            for i in range (400,800):
                for j in range (0,20):
                    sum += mask[j][i] #numpy arrays are y,x

            if sum == 0:
                print("SEEING BLACK 2:", sum)
                self.red = 2
            else:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)


        # self.red == 2 in between two red lines looking for second
        # drive straight until sum > 0
        # if sum > 0 then set self.red to 3 (saw the second red line)
        elif self.red == 2:
            sum = 0
            for i in range (400,800):
                sum += mask[20][i] #numpy arrays are y,x

            if sum == 0:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)
            else:
                print("SEEING RED 3:", sum)
                self.red = 3

        # self.red == 3 saw second red line
        # drive straight until black again
        # set self.red to 0 again ( we are past the crosswalk )
        # then go back to ur regular driving
        elif self.red == 3:
            sum = 0
            for i in range (400,800):
                sum += mask[20][i] #numpy arrays are y,x

            if sum == 0:
                print("SEEING BLACK 4", sum)
                self.red = 4
            else:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)

        elif self.red == 4:
            if (drive_error > 3) or (drive_error < -3):
                velocity.angular.z = d * drive_error
                #print("angular v:", velocity.angular.z)
                self.vel_pub.publish(velocity)
            else:
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)


def main(args):
    # nav = navigation()
    # rospy.init_node('navigation', anonymous=True)
    
    # rospy.Subscriber('right_follow', navigation.right_follow_callback, queue_size=1)
    # rospy.Subscriber('cross_walk', navigation.crosswalk_callback, queue_size=1)
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
    