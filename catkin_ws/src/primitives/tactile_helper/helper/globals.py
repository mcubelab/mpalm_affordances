from cv_bridge import CvBridge
import rospy
bridge = CvBridge()

def force_control_callback(msg):
    global delta_spring
    global delta_theta1
    global delta_theta2
    delta_spring = msg.data[0]
    delta_theta1 = msg.data[1]
    delta_theta2 = msg.data[2]

def initialize():
    global delta_spring
    global delta_theta1
    global delta_theta2
    global gel_img_l
    global gel_img_r
    delta_spring = 0
    delta_theta1 = 0
    delta_theta2 = 0
    gel_img_l = None
    gel_img_r = None

def callback_img_l(gel_msg):
    global gel_img_l
    gel_img_l = bridge.compressed_imgmsg_to_cv2(gel_msg, 'bgr8')

def callback_img_r(gel_msg):
    global gel_img_r
    gel_img_r = bridge.compressed_imgmsg_to_cv2(gel_msg, 'bgr8')