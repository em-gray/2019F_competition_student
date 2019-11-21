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

d = 4
speed = 1

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("R1/pi_camera/image_raw", Image, self.callback)
        self.vel_pub = rospy.Publisher("R1/cmd_vel", Twist, queue_size=30)
        self.prev_err = 0
        self.crosswalk = 0
        self.state = -3
        self.car_count = 0
        self.prev_sum = 0
        self.curr_sum = 0

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        height = np.size(frame, 0)
        width = np.size(frame,1)
        #print(height, width)

        #### frame processing for right line following
        roi_right = frame[height-300:height, width-640:width] #region of interest
        gray = cv.cvtColor(roi_right, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

        M = cv.moments(thresh)
        meep = (M["m00"])
        if meep < 1:
            meep = meep + 1
        cX = int(M["m10"] / meep)+640

        cv.circle(frame, (int(cX), height-100), 20, (0, 255, 0), -1)
        #cv.imshow("Cropped right corner", gray)

        drive_error = cX - 960
        #print("drive error:", drive_error)
        velocity = Twist()

        #### frame processing for red line detection
        roi_crosswalk = frame[height-200:height] # look at 50 rows of frame towards the bottom of screen
        lower_red = np.array([0,0,245])
        upper_red = np.array([10,10,255])
        mask_red = cv.inRange(roi_crosswalk, lower_red, upper_red)
        #cv.imshow("Crosswalk detection", mask_red)


        ########### STATE MACHINE #######################
        # inital turn to get to outer loop
        if (self.state == -3):
            #### frame processing for initial turn
            roi_init = frame[0:height, 0:width]
            lower_white = np.array([190,190,190])
            upper_white = np.array([255,255,255])
            mask_init = cv.inRange(roi_init, lower_white, upper_white)
            #cv.imshow("Initial Turn", mask_init)

            sum = 0
            for i in range (620, 660):
                for j in range (10, 500):
                    sum += mask_init[j][i] #numpy arrays are y,x

            if sum > 0:
                print("DRIVING STRAIGHT")
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)
            else:
                self.state = -2


        elif (self.state == -2):
            # turn left until good enough to line follow
            roi_init = frame[0:height, 0:width]
            lower_white = np.array([190,190,190])
            upper_white = np.array([255,255,255])
            mask_init = cv.inRange(roi_init, lower_white, upper_white)

            sum = 0
            for i in range (500, 780):
                for j in range (300, 400):
                    sum += mask_init[j][i] #numpy arrays are y,x

            #cv.rectangle(mask_init, (490, 290), (790, 390), (255,0,0), 1)
            #cv.rectangle(frame, (490, 290), (790, 390), (255,0,0), 1)
            if sum == 0:
                print("TURNING")
                velocity.angular.z = d
                self.vel_pub.publish(velocity)
            else:
                print("GOING TO REGULAR DRIVING")
                self.state = 0

        # regular following
        elif (self.state == 0) or (self.state == 5):
            if (drive_error > 3) or (drive_error < -3):
                velocity.angular.z = d * drive_error
                #print("angular v:", velocity.angular.z)
                self.vel_pub.publish(velocity)
            else:
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)

            ##### frame processing for car detection
            roi_blue = frame[0:height, 0:640]
            lower_b1 = np.array([100, 0, 0])
            upper_b1 = np.array([110, 5, 5])
            lower_b2 = np.array([115, 15, 15])
            upper_b2 = np.array([130, 25, 25])
            lower_b3 = np.array([170, 85, 85])
            upper_b3 = np.array([205, 105, 105])

            mask_b1 = cv.inRange(roi_blue, lower_b1, upper_b1)
            mask_b2 = cv.inRange(roi_blue, lower_b2, upper_b2)
            mask_b3 = cv.inRange(roi_blue, lower_b3, upper_b3)
            mask_blue_A = cv.bitwise_or(mask_b1, mask_b2)
            mask_blue_B = cv.bitwise_or(mask_b2, mask_b3)
            mask_blue = cv.bitwise_or(mask_blue_A, mask_blue_B)

            cv.rectangle(mask_blue, (145, 440), (235, 500), (255,255,255), 1)
            cv.imshow("Blue mask", mask_blue)

            for j in range (450,490):
                for i in range (150,230):
                    self.curr_sum = mask_blue[j][i]
            print("PREV:", self.prev_sum)
            print("CURR:", self.curr_sum)

            if (self.curr_sum > 0) and (self.prev_sum == 0):
                print("SNAPPED!")
                print("CAR SUM:", self.curr_sum)
                cv.imwrite("car%d.jpg" % self.car_count, frame)
                self.car_count += 1

            self.prev_sum = self.curr_sum


        #self.state == 0 havent seen anything
        # look for red
        # when u see red, set self.state to 1 (saw the red line transition to next step)
        # pause for 5 seconds
            sum = 0
            for i in range (400,800):
                sum += mask_red[20][i] #numpy arrays are y,x

             # SEE RED ONCE and STOP
            if sum > 0:
                print("SEEING RED 1:", sum)
                self.crosswalk += 1
                self.state = -1
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = 0
                self.vel_pub.publish(velocity)

        elif self.state == -1:
            # we look for pedestrian
            print("looking for ped")
            self.crosswalk += 1

            roi_ped = frame[0:500, 0:width]

            lower_white = np.array([190,190,190])
            upper_white = np.array([255,255,255])

            mask_white = cv.inRange(roi_ped, lower_white, upper_white)
            # cv.rectangle(mask_white, (545, 500-166), (800, 500-111), (255,0,0), 1)
            cv.rectangle(frame, (545, 500-166), (800, 500-111), (255,255,255), 1)
            #cv.imshow("Pedestrian detection", mask_white)

            sum_ped = 0
            for i in range (555,790):
                for j in range (500-156, 500-121):
                    sum_ped += mask_white[j][i]

            print("SUM PED:", sum_ped)

            if (sum_ped != 0): # if pedestrian, wait, and check frame again
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = 0
                self.vel_pub.publish(velocity)

            else:
                self.state = 1

        # self.state == 1 saw the first red line
        # need to wait for that region to be black again
        # drive forward untili sum =0
        # if sum = 0, set self.state to 2 (saw black transition to next)
        elif self.state == 1:
            sum = 0
            for i in range (400,800):
                for j in range (0,20):
                    sum += mask_red[j][i] #numpy arrays are y,x

            if sum == 0:
                print("SEEING BLACK 2:", sum)
                self.state = 2
            else:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)

        # self.state == 2 in between two red lines looking for second
        # drive straight until sum > 0
        # if sum > 0 then set self.state to 3 (saw the second red line)
        elif self.state == 2:
            sum = 0
            for i in range (400,800):
                sum += mask_red[20][i] #numpy arrays are y,x

            if sum == 0:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)
            else:
                print("SEEING RED 3:", sum)
                self.state = 3

        # self.state == 3 saw second red line
        # drive straight until black again
        # set self.state to 0 again ( we are past the crosswalk )
        # then go back to ur regular driving
        elif self.state == 3:
            sum = 0
            for i in range (400,800):
                sum += mask_red[20][i] #numpy arrays are y,x

            if sum == 0:
                print("SEEING BLACK 4", sum)
                self.state = 4
            else:
                velocity.angular.z = 0
                self.vel_pub.publish(velocity)
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)

        elif self.state == 4:
            if (drive_error > 3) or (drive_error < -3):
                #print("drive_error:", drive_error)
                velocity.angular.z = d * drive_error
                #print("angular v:", velocity.angular.z)
                self.vel_pub.publish(velocity)
            else:
                velocity.linear.x = speed
                self.vel_pub.publish(velocity)

            if drive_error > 7:
                print("SWITCHING TO STATE 5")
                self.state = 5

        cv.imshow("Robot Camera", frame)
        cv.waitKey(1)

        # if self.crosswalk == 3:
        #     velocity.angular.z = 0
        #     self.vel_pub.publish(velocity)
        #     velocity.linear.x = 0
        #     self.vel_pub.publish(velocity)

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
