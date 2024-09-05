#!/usr/bin/python3

import re
import rospy
import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
from time import sleep, time
from px4_mavros import Px4Controller
from gazebo_msgs.msg import ModelStates
from math import sin,cos,sqrt,atan2,acos,pi
from geometry_msgs.msg import Pose, Twist, TwistStamped

agents_state, targets_state = None, None
n, m = None, None
cmd_vel = Twist()
currP5 = np.zeros(3)
id_ = 5-5

def odom(msg):

	global agents_state, targets_state, n, m

	# agent_upper = 5
	# target_lower = 4
	agent_upper = 4
	target_lower = 3

	# Find all elements with the "uav" prefix and number less than 5
	agent_elements =\
				[element for element in msg.name if element.startswith('uav') and int(re.search(r'\d+', element).group()) < agent_upper]
	# print("agent_elements: ", agent_elements)

	# Sort the list based on the numeric part of the elements
	agent_elements_sorted = sorted(agent_elements, key=lambda x: int(re.search(r'\d+', x).group()))
	# print("agent_elements_sorted: ", agent_elements_sorted)

	agents_index = [msg.name.index(element) for element in agent_elements_sorted]

	# Find all elements with the "uav" prefix and number greater than 4
	target_elements =\
				[element for element in msg.name if element.startswith('uav') and int(re.search(r'\d+', element).group()) > target_lower]
	# print("target_elements: ", target_elements)

	# Sort the list based on the numeric part of the elements
	target_elements_sorted = sorted(target_elements, key=lambda x: int(re.search(r'\d+', x).group()))
	# print("target_elements_sorted: ", target_elements_sorted)

	target_index = [msg.name.index(element) for element in target_elements_sorted]

	n = len(agent_elements_sorted)
	m = len(target_elements_sorted)
	agents_state, targets_state = [], []
	# print("n, m:", n, m)

	for index in agents_index:

		P = np.array([msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z])
		
		dict_ = {"position": P}
		agents_state.append(dict_)
	# print("agents_state: ", agents_state)

	for index in target_index:

		P = np.array([msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z])
		V = np.array([msg.twist[index].linear.x, msg.twist[index].linear.y, msg.twist[index].linear.z])
		
		dict_ = {"position": P, "velocity": V}
		targets_state.append(dict_)
	# print("targets_state: ", targets_state)

def vel(msg):

	global currP5

	if abs(msg.twist.linear.x) > 1e-6 and  abs(msg.twist.linear.y) > 1e-6:

		currP5 = targets_state[id_]["position"]

def	controller(time_):

	global cmd_vel, currP5, targets_state, id_
	move_gain = 5
	delay_time = 1e-4

	u_des = np.array([1*((currP5[0] - targets_state[id_]["position"][0])),\
					1*((currP5[1] - targets_state[id_]["position"][1])),\
					9.8 - targets_state[id_]["position"][2]])

	cmd_vel.linear.x = 0.5*u_des[0]
	cmd_vel.linear.y = 0.5*u_des[1]
	cmd_vel.linear.z = u_des[2]

	px4.vel_control(cmd_vel)

virtual = np.array([6.5, 6.5])

def Path_Follower(t):

	global targets_state, id_

	u_des = np.array([0.0,0.0,0.0])

	if t < 20:

		if targets_state[id_]["position"][0] <= 18.0 and targets_state[id_]["position"][1] <= 6.5:

			u_des = np.array([10*(0.1), 2*(6.0-targets_state[id_]["position"][1]), 2*(9.8 - targets_state[id_]["position"][2])])

		if targets_state[id_]["position"][0] > 18.0 and targets_state[id_]["position"][1] <= 18.0:

			u_des = np.array([2*(18.5-targets_state[id_]["position"][0]), 10*(0.1), 9.8 - targets_state[id_]["position"][2]])

		if targets_state[id_]["position"][0] > 6.5 and targets_state[id_]["position"][1] > 18.0:

			u_des = np.array([-10*(0.1), 2*(18.5-targets_state[id_]["position"][1]), 9.8 - targets_state[id_]["position"][2]])

		if targets_state[id_]["position"][0] <= 6.5 and targets_state[id_]["position"][1] > 6.5:

			u_des = np.array([2*(6.0-targets_state[id_]["position"][0]), -10*(0.1), 9.8 - targets_state[id_]["position"][2]])

		u_mag = 0.45
	elif t < 35:
		u_des = 2.0*np.array([-10*(0.1), 0.0, 2*(9.8 - targets_state[id_]["position"][2])])
		u_mag = 0.5
	elif t < 50:
		u_des = 2.0*np.array([(targets_state[2]["position"][0]-targets_state[id_]["position"][0]) + 1.0 + targets_state[2]["velocity"][0],\
							(targets_state[2]["position"][1]-targets_state[id_]["position"][1]) + 1.0 + targets_state[2]["velocity"][1],\
							2*(9.8 - targets_state[id_]["position"][2])])
		u_mag = 0.5
	else:

		# u_des = np.array([0.5*((P7[0]-P8[0]) + 0.0 + P7v[0]) + 0.5*((P6[0]-P8[0]) - 1.1) + 0.5*((P9[0]-P8[0]) - 1.1),\
		# 				0.5*((P7[1]-P8[1]) - 1.1 + P7v[1]) + 0.5*((P6[1]-P8[1]) - 1.1) + 0.5*((P9[1]-P8[1]) + 0.0),\
		# 				2*(9.8 - P8[2])])
		u_des = np.array([0.5*((targets_state[2]["position"][0]-targets_state[id_]["position"][0]) + 0.0 + targets_state[2]["velocity"][0]),\
						0.5*((targets_state[2]["position"][1]-targets_state[id_]["position"][1]) + 1.1 + targets_state[2]["velocity"][1]),\
						2*(9.8 - targets_state[id_]["position"][2])])

		u_mag = 0.5

	u_dir = u_des/np.linalg.norm(u_des)

	print("P5: ", targets_state[id_]["position"])
	print("P5 u_mag: ", u_mag)
	print("P5 u_dir: ", u_dir)

	cmd_vel.linear.x = 1.0*u_mag*u_dir[0]
	cmd_vel.linear.y = 1.0*u_mag*u_dir[1]
	cmd_vel.linear.z = u_des[2]
	px4.vel_control(cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_5')
		px4 = Px4Controller("uav5")
		rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
		rospy.Subscriber('/uav5/cmd_vel', TwistStamped, vel, queue_size=10)
		rate = rospy.Rate(100)

		while (targets_state is None) or (n is None) or (m is None):

			rate.sleep()

		currP5 = targets_state[id_]["position"]

		last = time()

		while not rospy.is_shutdown():

			controller(np.round(time() - last, 2))
			
			print("Time Now: ", np.round(time() - last, 2))
			# Path_Follower(np.round(time() - last, 2))

			rate.sleep()
	except rospy.ROSInterruptException:

		pass
