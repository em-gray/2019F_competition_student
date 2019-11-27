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

        # height = np.size(frame, 0)
        # width = np.size(frame,1)

        thresh_r = self.thresh(frame, "right")
        M_r = cv.moments(thresh_r)
        meep = (M_r["m00"])
        if meep < 1:
            meep = meep + 1
        cX_r = int(M_r["m10"] / meep)+640
        drive_err_r = cX_r - 960

        ########### STATE MACHINE #######################
        # inital turn to get to outer loop
        if (self.state == -3):
            mask_straight = self.filter_w(frame, "straight")
            sum = np.sum(mask_straight)

            if sum > 0:
                print("DRIVING STRAIGHT")
                self.velocities(0,speed)
            else:
                self.state = -2

        elif (self.state == -2):
            mask_turn = self.filter_w(frame, "turn")
            sum = np.sum(mask_turn)
            if sum == 0:
                print("TURNING")
                self.velocities(d, None)
            else:
                print("GOING TO REGULAR DRIVING")
                self.state = 0

        # regular following
        elif (self.state == 0) or (self.state == 5):
            if (drive_err_r > 3) or (drive_err_r < -3):
                self.velocities(d*drive_err_r, None)
            else:
                self.velocities(None, speed)

            mask_blue = self.filter_b(frame, "left")
            self.curr_sum = np.sum(mask_blue)

            if (self.curr_sum > 0) and (self.prev_sum == 0):
                print("SNAPPED!")
                print("CAR SUM:", self.curr_sum)
                cv.imwrite("car%d.jpg" % self.car_count, frame)
                self.car_count += 1

            self.prev_sum = self.curr_sum

        #self.state == 0 havent seen anything
        # look for red
        # when u see red, set self.state to 1 (saw the red line transition to next step)
            mask_red = self.filter_r(frame,"red")
            sum = np.sum(mask_red)

            if sum > 0:
                print("SEEING RED 1:", sum)
                self.state = -1
                self.velocities(0, 0)

        elif self.state == -1:
            # we look for pedestrian
            print("looking for ped")

            mask_ped = self.filter_w(frame, "pedestrian")
            sum_ped = np.sum(mask_ped)

            print("SUM PED:", sum_ped)

            if (sum_ped != 0): # if pedestrian, wait, and check frame again
                self.velocities(0, 0)
            else:
                self.state = 1

        # self.state == 1 saw the first red line
        # need to wait for that region to be black again
        # drive forward untili sum =0
        # if sum = 0, set self.state to 2 (saw black transition to next)
        elif self.state == 1:
            mask_red = self.filter_r(frame,"black")
            sum = np.sum(mask_red)

            if sum == 0:
                print("SEEING BLACK 2:", sum)
                self.state = 2
            else:
                self.velocities(0, speed)

        # self.state == 2 in between two red lines looking for second
        # drive straight until sum > 0
        # if sum > 0 then set self.state to 3 (saw the second red line)
        elif self.state == 2:
            mask_red = self.filter_r(frame,"red")
            sum = np.sum(mask_red)

            if sum == 0:
                self.velocities(0, speed)
            else:
                print("SEEING RED 3:", sum)
                self.state = 3

        # self.state == 3 saw second red line
        # drive straight until black again
        # set self.state to 0 again ( we are past the crosswalk )
        # then go back to ur regular driving
        elif self.state == 3:
            mask_red = self.filter_r(frame,"red")
            sum = np.sum(mask_red)

            if sum == 0:
                print("SEEING BLACK 4", sum)
                self.state = 4
            else:
                self.velocities(0, speed)

        elif self.state == 4:
            if (drive_err_r > 3) or (drive_err_r < -3):
                self.velocities(d*drive_err_r, None)
            else:
                self.velocities(None, speed)

            if drive_err_r > 7:
                if self.crosswalk == 1:
                    print("SWITCHING TO STATE 6")
                    self.state = 6
                else:
                    print("SWITCHING TO STATE 5")
                    self.crosswalk += 1
                    print("crosswalks:", self.crosswalk)
                    self.state = 5

        elif self.state == 6:
            if (drive_err_r > 3) or (drive_err_r < -3):
                self.velocities(d*drive_err_r, None)
            else:
                self.velocities(None, speed)

            mask_blue = self.filter_b(frame, "left")
            self.curr_sum = np.sum(mask_blue)

            if (self.curr_sum == 0) and (self.prev_sum > 0):
                print("SWITCH TO LEFT FOLLOW")
                self.state = 7

            self.prev_sum = self.curr_sum

        elif self.state == 7:
            # thresh_l = self.thresh(frame, "left")
            # M_l = cv.moments(thresh_l)
            # meep = (M_l["m00"])
            # if meep < 1:
            #     meep = meep + 1
            # cX_l = int(M_l["m10"] / meep)
            # drive_err_l = cX_l - 320

            print("IN STATE 7")

            mask_w_2_in = self.filter_w(frame, "inner")
            mask_blue = self.filter_b(frame, "inner")
            mask_com = cv.bitwise_or(mask_w_2_in, mask_blue)
            sum = np.sum(mask_com)

            print("LEFT TURN:", sum)

            if sum == 0:
                self.state = 8
            else:
                if (drive_err_r > 3) or (drive_err_r < -3):
                    self.velocities(d*drive_err_r, None)
                else:
                    self.velocities(None, speed)

        elif self.state == 8:
            print("STATE 8")

            # mask = self.filter_w(frame, "turn in")
            # sum = np.sum(mask)

            self.velocities(0, 0)

        self.present(frame, cX_r)

    # function to take care of publishing velocities
    def velocities(self, ang, lin):
        velocity = Twist()
        if ang != None:
            velocity.angular.z = ang
            self.vel_pub.publish(velocity)
        if lin != None:
            velocity.linear.x = lin
            self.vel_pub.publish(velocity)
        return

    def present(self, frame, cX_r):
        cv.rectangle(frame, (215, 485), (230, 490), (255,255,255), 1) # car detection
        cv.rectangle(frame, (500, 334), (800, 389), (255,255,255), 2) # pedestrian detection
        cv.rectangle(frame, (5, 550), (350, 650), (255,255,255), 1)
        cv.circle(frame, (int(cX_r), 620), 20, (0, 255, 0), -1) # right white line centroid
        cv.imshow("Robot Camera", frame)
        cv.waitKey(1)
        return

    def thresh(self, frame, side):
        if side == "right":
            roi = frame[420:720, 640:1280]
        elif side == "left":
            roi = frame[550:650, 0:640]
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
        return thresh

    def filter_w(self, frame, purpose):
        if purpose == "straight":
            roi = frame[10:500,620:660]
        elif purpose == "turn":
            roi = frame[300:400,500:780]
        elif purpose == "pedestrian":
            roi = frame[334:389,500:800]
        elif purpose == "inner":
            roi = frame[550:650, 5:350]
        elif purpose == "turn in":
            roi = frame[400:650,400:800]
        lower_white = np.array([190,190,190])
        upper_white = np.array([255,255,255])
        mask = cv.inRange(roi, lower_white, upper_white)
        return mask

    def filter_r(self, frame, purpose):
        if purpose == "red":
            roi = frame[520:522,400:800]
        elif purpose == "black":
            roi = frame[520:540,400:800]
        lower_red = np.array([0,0,245])
        upper_red = np.array([10,10,255])
        mask = cv.inRange(roi, lower_red, upper_red)
        return mask

    def filter_b(self, frame, side):
        if side == "left":
            roi = frame[485:490, 215:230]
        elif side == "right":
            roi = frame[0:720, 640:1280]
        elif side == "inner":
            roi = frame[550:650, 5:350]
        lower_b1 = np.array([100, 0, 0])
        upper_b1 = np.array([110, 5, 5])
        lower_b2 = np.array([115, 15, 15])
        upper_b2 = np.array([130, 25, 25])
        lower_b3 = np.array([170, 85, 85])
        upper_b3 = np.array([205, 105, 105])
        mask_b1 = cv.inRange(roi, lower_b1, upper_b1)
        mask_b2 = cv.inRange(roi, lower_b2, upper_b2)
        mask_b3 = cv.inRange(roi, lower_b3, upper_b3)
        mask_blue_A = cv.bitwise_or(mask_b1, mask_b2)
        mask_blue_B = cv.bitwise_or(mask_b2, mask_b3)
        mask_blue = cv.bitwise_or(mask_blue_A, mask_blue_B)
        return mask_blue

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
