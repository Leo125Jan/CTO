#!/usr/bin/python3

import time
import rospy
import numpy as np
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from pyquaternion import Quaternion
from std_msgs.msg import Float64MultiArray
from mavros_msgs.msg import State, ParamValue
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped, Twist
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, ParamSet


class Px4Controller:

    def __init__(self, uavtype):

        # Variable - Nominal
        self.type = uavtype
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.height = np.array([50.0, 50.0, 50.0])
        self.loiter_RAD = np.array([150.0, 150.0, 150.0])

        # Variable - Custom
        self.imu = None
        self.current_state = None
        self.current_heading = None
        self.arm_received = False
        self.offboard_received = False
        self.takeoff_received = False
        self.loiter_received = False

        # Variable - Service
        self.arm_state = False
        self.offboard_state = False
        self.cmd_takeoff = CommandTOL()

        # ROS Subscriber & Publisher
        self.state_sub = rospy.Subscriber(uavtype + "/mavros/state", State, self.state_cb, queue_size=100)
        self.imu_sub = rospy.Subscriber(uavtype + "/mavros/imu/data", Imu, self.imu_callback, queue_size=100)
        self.odom_sub = rospy.Subscriber(uavtype + "/mavros/local_position/odom", Odometry, self.odom_cb, queue_size=100)

        self.pos_pub = rospy.Publisher(uavtype + '/mavros/setpoint_position/local', PoseStamped, queue_size=100)
        self.vel_pub = rospy.Publisher(uavtype + '/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)

        # ROS Service
        self.paramService = rospy.ServiceProxy(uavtype + '/mavros/param/set', ParamSet)
        self.armService = rospy.ServiceProxy(uavtype + '/mavros/cmd/arming', CommandBool)
        self.flightModeService = rospy.ServiceProxy(uavtype + '/mavros/set_mode', SetMode)
        self.takeoffService = rospy.ServiceProxy(uavtype + '/mavros/cmd/takeoff', CommandTOL)

        print(uavtype + " Px4 Controller Initialized!")

    def state_cb(self, msg):

        self.current_state = msg

    def odom_cb(self, data):

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.z = data.pose.pose.position.z

    def imu_callback(self, msg):

        self.imu = msg
        self.current_heading = self.q2yaw(self.imu.orientation)

    def q2yaw(self, q):

        if isinstance(q, Quaternion):

            rotate_z_rad = q.yaw_pitch_roll[0]
        else:

            q_ = Quaternion(q.w, q.x, q.y, q.z)
            rotate_z_rad = q_.yaw_pitch_roll[0]

        return rotate_z_rad
 
    def Arm(self):

        ret = self.armService(True)

        if ret.success:

            print(self.type + "_arm: succeed")
            self.arm_received = True

            return True
        else:

            print(self.type + "_arm: fail")
            return False

    def Offboard(self):

        ret = self.flightModeService(custom_mode='OFFBOARD')

        if ret.mode_sent:

            print(self.type +"_offboard: succeed")
            self.offboard_received = True

            return True
        else:

            print(self.type +"_offboard: fail")
            return False

    def takeoff(self):

        ret1 = self.paramService(param_id = "MIS_TAKEOFF_ALT", value = ParamValue(integer = 0, real = self.height[int(self.type[-1])]))
        ret2 = self.takeoffService(min_pitch = 0.0, yaw = 0.0, latitude = 0.0, longitude = 0.0, altitude = self.height[int(self.type[-1])])
        
        if ret1.success:

            print(self.type + "_takeoff: succeed")
            self.takeoff_received = True

            return True
        else:

            print(self.type + "_takeoff: fail")
            return False

    def hold(self):

        if self.z >= self.height[int(self.type[-1])]-5:

            ret1 = self.paramService(param_id = "NAV_LOITER_RAD", value = ParamValue(integer = 0, real = self.loiter_RAD[int(self.type[-1])]))
            ret2 = self.flightModeService(custom_mode='AUTO.LOITER')
            ret3 = self.flightModeService(custom_mode='OFFBOARD')

            if ret2.mode_sent:

                print(self.type + "_loiter: succeed")
                self.loiter_received = True

                return True
            else:

                print(self.type + "_loiter: fail")
                return False
            
if __name__ == '__main__':

    try:
        rospy.init_node('ArmandOffboard')

        uavtype = ["uav0","uav1","uav2"]
        px3_1 = Px4Controller(uavtype[0])
        px3_2 = Px4Controller(uavtype[1])
        px3_3 = Px4Controller(uavtype[2])

        last_time_1 = rospy.Time.now()
        last_time_2 = rospy.Time.now()
        last_time_3 = rospy.Time.now()

        while not rospy.is_shutdown():

            if rospy.Time.now() - last_time_1 > rospy.Duration(10):

                if not px3_1.current_state.armed or px3_1.current_state.mode != "OFFBOARD":

                    # if not px3_1.arm_received or not px3_1.offboard_received:

                    px3_1.Arm()
                    px3_1.Offboard()

                    last_time_1 = rospy.Time.now()
                
                if not px3_1.takeoff_received:

                    px3_1.takeoff()

                if not px3_1.loiter_received:

                    px3_1.hold()

            if rospy.Time.now() - last_time_2 > rospy.Duration(10):

                if not px3_2.current_state.armed or px3_2.current_state.mode != "OFFBOARD":

                    if not px3_2.arm_received or not px3_2.offboard_received:

                        px3_2.Arm()
                        px3_2.Offboard()

                    last_time_2 = rospy.Time.now()
                
                if not px3_2.takeoff_received:

                    px3_2.takeoff()

                if not px3_2.loiter_received:

                    px3_2.hold()

            if rospy.Time.now() - last_time_3 > rospy.Duration(10):

                if not px3_3.current_state.armed or px3_3.current_state.mode != "OFFBOARD":

                    if not px3_3.arm_received or not px3_3.offboard_received:

                        px3_3.Arm()
                        px3_3.Offboard()

                    last_time_3 = rospy.Time.now()
                
                if not px3_3.takeoff_received:

                    px3_3.takeoff()

                if not px3_3.loiter_received:

                    px3_3.hold()

    except rospy.ROSInterruptException:
        pass 

