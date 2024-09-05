#!/usr/bin/python3

import rospy
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from pyquaternion import Quaternion
from px4_mavros import Px4Controller
from gazebo_msgs.msg import ModelStates
from math import sin,cos,sqrt,atan2,acos,pi

P0, P1, P2, P3, P3_v = None, None, None, None, None
current_heading = 0.0
cmd_vel = Twist()

def odom(msg):

	global P0, P1, P2, P3, P3_v
	
	UAV0_index = msg.name.index('uav0')
	UAV1_index = msg.name.index('uav1')
	UAV2_index = msg.name.index('uav2')
	UAV3_index = msg.name.index('uav3')

	P0 = np.array([msg.pose[UAV0_index].position.x, msg.pose[UAV0_index].position.y, msg.pose[UAV0_index].position.z])
	P1 = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P2 = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
	P3 = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
	P3_v = np.array([msg.twist[UAV3_index].linear.x, msg.twist[UAV3_index].linear.y, msg.twist[UAV3_index].linear.z])

def imu_callback(msg):

	global current_heading

	current_heading = q2yaw(msg.orientation)

def q2yaw(q):

	if isinstance(q, Quaternion):

		rotate_z_rad = q.yaw_pitch_roll[0]
	else:

		q_ = Quaternion(q.w, q.x, q.y, q.z)
		rotate_z_rad = q_.yaw_pitch_roll[0]

	return rotate_z_rad 
	
def	controller():

	global cmd_vel

	# Position Control
	cmd_vel.linear.x = 0.5*((P1[0] - P2[0]) + 4.0 + (P0[0] - P2[0]) - 4.0) + 0.5*(P3[0] - P2[0]) + 1*P3_v[0]
	cmd_vel.linear.y = 0.5*((P1[1] - P2[1]) + 4.0*sqrt(3) + (P0[1] - P2[1]) + 4.0*sqrt(3)) + 0.5*(P3[1] - P2[1] + 2.0*sqrt(3)) + 1*P3_v[1]
	cmd_vel.linear.z = 1.0 - P2[2]

	# Heading Control
	theta = current_heading
	perspective = np.array([1*cos(theta), 1*sin(theta)])

	perspective_t = np.array([P3[0] - P2[0], P3[1] - P2[1]]);
	perspective_t /= np.linalg.norm(perspective_t)

	axis = np.cross(perspective, perspective_t)
	axis = axis/np.linalg.norm(axis)

	dot_product = np.dot(perspective, perspective_t)
	dtheta = np.arccos( dot_product/(np.linalg.norm(perspective) * np.linalg.norm(perspective_t)) )

	cmd_vel.angular.z = dtheta*axis

	px4_2.vel_control(cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_2')
		px4_2 = Px4Controller("uav2")
		pose_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size = 10)
		imu_sub = rospy.Subscriber("/uav2/mavros/imu/data", Imu, imu_callback, queue_size = 10)

		rate = rospy.Rate(100)

		while P2 is None:

			rate.sleep()

		t = 0

		while not rospy.is_shutdown():

			controller()

			t += 0.0005
			rate.sleep()

	except rospy.ROSInterruptException:

		pass