#!/usr/bin/python3

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import re
import sys
import rospy
import random
import numpy as np
import skfuzzy as fuzz
from time import sleep, time
from sensor_msgs.msg import Imu
from scipy.spatial import distance
from pyquaternion import Quaternion
from px4_mavros import Px4Controller
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull, Delaunay
from gazebo_msgs.msg import ModelStates, LinkStates
from geometry_msgs.msg import Pose, Twist, TwistStamped
from scipy.optimize import linear_sum_assignment, linprog
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension

class QGC():

	def __init__(self):

		# Publisher & Subscriber
		self.states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.states_callback, queue_size = 100, buff_size = 52428800)
		self.link_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_callback, queue_size = 100)

		self.UAV_StateI = rospy.Publisher("/uav/StateI", ModelStates, queue_size = 100)
		self.UAV_LinkI = rospy.Publisher("/uav/LinkI", LinkStates, queue_size = 100)
		# self.UAV0_Ref = rospy.Publisher("/uav0/Ref", Float64MultiArray, queue_size = 100)
		# self.UAV1_Ref = rospy.Publisher("/uav1/Ref", Float64MultiArray, queue_size = 100)
		# self.UAV2_Ref = rospy.Publisher("/uav2/Ref", Float64MultiArray, queue_size = 100)
		# self.UAV3_Ref = rospy.Publisher("/uav3/Ref", Float64MultiArray, queue_size = 100)
		# self.UAV4_Ref = rospy.Publisher("/uav4/Ref", Float64MultiArray, queue_size = 100)
		self.agent_Ref = None

		# self.UAV5_cmd_vel = rospy.Publisher('/uav5/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV5_hold_vel = rospy.Publisher('/uav5/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV6_cmd_vel = rospy.Publisher('/uav6/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV6_hold_vel = rospy.Publisher('/uav6/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV7_cmd_vel = rospy.Publisher('/uav7/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV7_hold_vel = rospy.Publisher('/uav7/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV8_cmd_vel = rospy.Publisher('/uav8/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV8_hold_vel = rospy.Publisher('/uav8/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV9_cmd_vel = rospy.Publisher('/uav9/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=100)
		# self.UAV9_hold_vel = rospy.Publisher('/uav9/cmd_vel', TwistStamped, queue_size=100)
		self.target_cmd_vel = None
		self.target_hold_vel = None

		# Variable of Control
		# self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o = None, None, None, None, None, None
		# self.P3, self.P3o, self.P4, self.P4o = None, None, None, None

		# self.P5, self.P5v, self.P6, self.P6v, self.P7, self.P7v = None, None, None, None, None, None
		# self.P8, self.P8v, self.P9, self.P9v = None, None, None, None

		# self.UAV0_currH, self.UAV1_currH, self.UAV2_currH, self.UAV3_currH, self.UAV4_currH = None, None, None, None, None
		# self.P0s, self.P1s, self.P2s, self.P3s, self.P4s = None, None, None, None, None
		# self.Perp0, self.Perp1, self.Perp2, self.Perp3, self.Perp4 = None, None, None, None, None
		# self.Cam0, self.Cam1, self.Cam2, self.Cam3, self.Cam4 = None, None, None, None, None
		# self.Cam0o, self.Cam1o, self.Cam2o, self.Cam3o, self.Cam4o = None, None, None, None, None

		self.agents_state = None
		self.agents_name = None
		self.targets_state = None
		self.targets_name = None
		self.UAV_currH = None
		self.Cams_state = None
		self.Cams_name = None
		self.n, self.m = None, None

		# Target Velocity
		self.velocities = (np.random.rand(5, 2) - 0.5)  # Random initial velocities (-0.5 to 0.5)

	def states_callback(self, msg):

		# UAVs_index = [msg.name.index('uav0'), msg.name.index('uav1'), msg.name.index('uav2'), msg.name.index('uav3'),\
		# 	msg.name.index('uav4'), msg.name.index('uav5'), msg.name.index('uav6'), msg.name.index('uav7'),\
		# 	msg.name.index('uav8'), msg.name.index('uav9')]

		# P0 = np.array([msg.pose[UAVs_index[0]].position.x, msg.pose[UAVs_index[0]].position.y, msg.pose[UAVs_index[0]].position.z])
		# P0o = np.array([msg.pose[UAVs_index[0]].orientation.x, msg.pose[UAVs_index[0]].orientation.y,\
		# 				msg.pose[UAVs_index[0]].orientation.z, msg.pose[UAVs_index[0]].orientation.w])
		# P1 = np.array([msg.pose[UAVs_index[1]].position.x, msg.pose[UAVs_index[1]].position.y, msg.pose[UAVs_index[1]].position.z])
		# P1o = np.array([msg.pose[UAVs_index[1]].orientation.x, msg.pose[UAVs_index[1]].orientation.y,\
		# 				msg.pose[UAVs_index[1]].orientation.z, msg.pose[UAVs_index[1]].orientation.w])
		# P2 = np.array([msg.pose[UAVs_index[2]].position.x, msg.pose[UAVs_index[2]].position.y, msg.pose[UAVs_index[2]].position.z])
		# P2o = np.array([msg.pose[UAVs_index[2]].orientation.x, msg.pose[UAVs_index[2]].orientation.y,\
		# 				msg.pose[UAVs_index[2]].orientation.z, msg.pose[UAVs_index[2]].orientation.w])
		# P3 = np.array([msg.pose[UAVs_index[3]].position.x, msg.pose[UAVs_index[3]].position.y, msg.pose[UAVs_index[3]].position.z])
		# P3o = np.array([msg.pose[UAVs_index[3]].orientation.x, msg.pose[UAVs_index[3]].orientation.y,\
		# 				msg.pose[UAVs_index[3]].orientation.z, msg.pose[UAVs_index[3]].orientation.w])
		# P4 = np.array([msg.pose[UAVs_index[4]].position.x, msg.pose[UAVs_index[4]].position.y, msg.pose[UAVs_index[4]].position.z])
		# P4o = np.array([msg.pose[UAVs_index[4]].orientation.x, msg.pose[UAVs_index[4]].orientation.y,\
		# 				msg.pose[UAVs_index[4]].orientation.z, msg.pose[UAVs_index[4]].orientation.w])

		# P5 = np.array([msg.pose[UAVs_index[5]].position.x, msg.pose[UAVs_index[5]].position.y, msg.pose[UAVs_index[5]].position.z])
		# P5v = np.array([msg.twist[UAVs_index[5]].linear.x, msg.twist[UAVs_index[5]].linear.y, msg.twist[UAVs_index[5]].linear.z])
		# P6 = np.array([msg.pose[UAVs_index[6]].position.x, msg.pose[UAVs_index[6]].position.y, msg.pose[UAVs_index[6]].position.z])
		# P6v = np.array([msg.twist[UAVs_index[6]].linear.x, msg.twist[UAVs_index[6]].linear.y, msg.twist[UAVs_index[6]].linear.z])
		# P7 = np.array([msg.pose[UAVs_index[7]].position.x, msg.pose[UAVs_index[7]].position.y, msg.pose[UAVs_index[7]].position.z])
		# P7v = np.array([msg.twist[UAVs_index[7]].linear.x, msg.twist[UAVs_index[7]].linear.y, msg.twist[UAVs_index[7]].linear.z])
		# P8 = np.array([msg.pose[UAVs_index[8]].position.x, msg.pose[UAVs_index[8]].position.y, msg.pose[UAVs_index[8]].position.z])
		# P8v = np.array([msg.twist[UAVs_index[8]].linear.x, msg.twist[UAVs_index[8]].linear.y, msg.twist[UAVs_index[8]].linear.z])
		# P9 = np.array([msg.pose[UAVs_index[9]].position.x, msg.pose[UAVs_index[9]].position.y, msg.pose[UAVs_index[9]].position.z])
		# P9v = np.array([msg.twist[UAVs_index[9]].linear.x, msg.twist[UAVs_index[9]].linear.y, msg.twist[UAVs_index[9]].linear.z])

		# self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o = P0, P0o, P1, P1o, P2, P2o
		# self.P3, self.P3o, self.P4, self.P4o = P3, P3o, P4, P4o
		
		# self.P5, self.P5v, self.P6, self.P6v, self.P7, self.P7v = P5, P5v, P6, P6v, P7, P7v
		# self.P8, self.P8v, self.P9, self.P9v = P8, P8v, P9, P9v

		# self.UAV0_currH = self.q2yaw(self.P0o)
		# self.UAV1_currH = self.q2yaw(self.P1o)
		# self.UAV2_currH = self.q2yaw(self.P2o)
		# self.UAV3_currH = self.q2yaw(self.P3o)
		# self.UAV4_currH = self.q2yaw(self.P4o)

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
		self.agents_name = agent_elements_sorted

		agents_index = [msg.name.index(element) for element in agent_elements_sorted]

		# Find all elements with the "uav" prefix and number greater than 4
		target_elements =\
					[element for element in msg.name if element.startswith('uav') and int(re.search(r'\d+', element).group()) > target_lower]
		# print("target_elements: ", target_elements)

		# Sort the list based on the numeric part of the elements
		target_elements_sorted = sorted(target_elements, key=lambda x: int(re.search(r'\d+', x).group()))
		# print("target_elements_sorted: ", target_elements_sorted)
		self.targets_name = target_elements_sorted

		target_index = [msg.name.index(element) for element in target_elements_sorted]

		# Number of agnet and target
		n = len(agent_elements_sorted)
		m = len(target_elements_sorted)
		agents_state, targets_state = [], []
		# print("n, m:", n, m)

		for index in agents_index:

			P = np.array([msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z])
			O = np.array([msg.pose[index].orientation.x, msg.pose[index].orientation.y,\
							msg.pose[index].orientation.z, msg.pose[index].orientation.w])
			
			dict_ = {"position": P, "orientation": O}
			agents_state.append(dict_)
		# print("agents_state: ", agents_state)

		for index in target_index:

			P = np.array([msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z])
			V = np.array([msg.twist[index].linear.x, msg.twist[index].linear.y, msg.twist[index].linear.z])
			
			dict_ = {"position": P, "velocity": V}
			targets_state.append(dict_)
		# print("targets_state: ", targets_state)

		UAV_currH = [self.q2yaw(state["orientation"]) for state in agents_state]
		# print("UAV_currH: ", UAV_currH)

		self.agents_state = agents_state
		self.targets_state = targets_state
		self.n, self.m = n, m
		self.UAV_currH = UAV_currH

	def link_callback(self, msg):

		# Link_index = [msg.name.index('uav0::cgo3_camera_link'), msg.name.index('uav1::cgo3_camera_link'),\
		# 				msg.name.index('uav2::cgo3_camera_link'), msg.name.index('uav3::cgo3_camera_link'),\
		# 				msg.name.index('uav4::cgo3_camera_link')]

		# Cam0 = np.array([msg.pose[Link_index[0]].position.x, msg.pose[Link_index[0]].position.y, msg.pose[Link_index[0]].position.z])
		# Cam0o = np.array([msg.pose[Link_index[0]].orientation.x, msg.pose[Link_index[0]].orientation.y,\
		# 				msg.pose[Link_index[0]].orientation.z, msg.pose[Link_index[0]].orientation.w])

		# Cam1 = np.array([msg.pose[Link_index[1]].position.x, msg.pose[Link_index[1]].position.y, msg.pose[Link_index[1]].position.z])
		# Cam1o = np.array([msg.pose[Link_index[1]].orientation.x, msg.pose[Link_index[1]].orientation.y,\
		# 				msg.pose[Link_index[1]].orientation.z, msg.pose[Link_index[1]].orientation.w])

		# Cam2 = np.array([msg.pose[Link_index[2]].position.x, msg.pose[Link_index[2]].position.y, msg.pose[Link_index[2]].position.z])
		# Cam2o = np.array([msg.pose[Link_index[2]].orientation.x, msg.pose[Link_index[2]].orientation.y,\
		# 				msg.pose[Link_index[2]].orientation.z, msg.pose[Link_index[2]].orientation.w])

		# Cam3 = np.array([msg.pose[Link_index[3]].position.x, msg.pose[Link_index[3]].position.y, msg.pose[Link_index[3]].position.z])
		# Cam3o = np.array([msg.pose[Link_index[3]].orientation.x, msg.pose[Link_index[3]].orientation.y,\
		# 				msg.pose[Link_index[3]].orientation.z, msg.pose[Link_index[3]].orientation.w])

		# Cam4 = np.array([msg.pose[Link_index[4]].position.x, msg.pose[Link_index[4]].position.y, msg.pose[Link_index[4]].position.z])
		# Cam4o = np.array([msg.pose[Link_index[4]].orientation.x, msg.pose[Link_index[4]].orientation.y,\
		# 				msg.pose[Link_index[4]].orientation.z, msg.pose[Link_index[4]].orientation.w])

		# self.Cam0, self.Cam1, self.Cam2, self.Cam3, self.Cam4 = Cam0, Cam1, Cam2, Cam3, Cam4
		# self.Cam0o, self.Cam1o, self.Cam2o, self.Cam3o, self.Cam4o = Cam0o, Cam1o, Cam2o, Cam3o, Cam4o


		# Extract elements with 'uav#::cgo3_camera_link'
		cams_element = [element for element in msg.name if re.match(r'uav\d+::cgo3_camera_link', element)]

		# Sort the list based on the numeric part
		cams_element_sorted = sorted(cams_element, key=lambda x: int(re.search(r'uav(\d+)', x).group(1)))
		# print("cams_element_sorted: ", cams_element_sorted)
		self.Cams_name = cams_element_sorted

		cams_index = [msg.name.index(element) for element in cams_element_sorted]
		cams_state = []

		for index in cams_index:

			P = np.array([msg.pose[index].position.x, msg.pose[index].position.y, msg.pose[index].position.z])
			O = np.array([msg.pose[index].orientation.x, msg.pose[index].orientation.y,\
							msg.pose[index].orientation.z, msg.pose[index].orientation.w])
			
			dict_ = {"position": P, "orientation": O}
			cams_state.append(dict_)
		# print("cams_state: ", cams_state)

		self.Cams_state = cams_state

	def UAV_Sweet_Spot(self):

		# print("UAV0_currH, UAV1_currH, UAV2_currH: ", self.UAV0_currH, self.UAV1_currH, self.UAV2_currH)

		# theta0, theta1, theta2, theta3, theta4 = self.UAV0_currH, self.UAV1_currH, self.UAV2_currH, self.UAV3_currH, self.UAV4_currH
		# self.Perp0, self.Perp1, self.Perp2, self.Perp3, self.Perp4 = \
		# np.array([1*cos(theta0), 1*sin(theta0)]), np.array([1*cos(theta1), 1*sin(theta1)]), np.array([1*cos(theta2), 1*sin(theta2)]),\
		# np.array([1*cos(theta3), 1*sin(theta3)]), np.array([1*cos(theta4), 1*sin(theta4)])

		# self.P0s = self.P0[0:2] + 4.0*np.cos(20)*self.Perp0
		# self.P1s = self.P1[0:2] + 4.0*np.cos(20)*self.Perp1
		# self.P2s = self.P2[0:2] + 4.0*np.cos(20)*self.Perp2
		# self.P3s = self.P3[0:2] + 4.0*np.cos(20)*self.Perp3
		# self.P4s = self.P4[0:2] + 4.0*np.cos(20)*self.Perp4
		# print("P0s, P1s, P2s: ", self.P0s, self.P1s, self.P2s)

		self.Ps = [state["position"][0:2]\
					+ 4.0*np.cos(20)*np.array([1*cos(self.q2yaw(state["orientation"])), 1*sin(self.q2yaw(state["orientation"]))])\
					for state in self.agents_state]
		# print("Ps: ", self.Ps)

	def q2yaw(self, q):

		if isinstance(q, Quaternion):

			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q[3], q[0], q[1], q[2])
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad

	def Agglomerative_Hierarchical_Clustering(self):

		# Sample data points
		# data_points = np.array([self.P5[0:2], self.P6[0:2], self.P7[0:2], self.P8[0:2], self.P9[0:2]])
		data_points = np.array([state["position"][0:2] for state in self.targets_state])

		# Custom distance threshold for merging clusters
		threshold = 4.0  # Adjust as needed
		# threshold = 2.5  # Adjust as needed
		# threshold = 3.0*cameras[0].incircle_r  # Adjust as needed
		# print("threshold: ", threshold)

		# Initialize each data point as its own cluster
		clusters = [[point] for point in data_points]
		cluster_mapping = {index_: [index_] for (index_, element) in enumerate(data_points)}
		# cluster_mapping_save = {str(0): [] for (index_, element) in enumerate(data_points)}
		cluster_mapping_save = {0: [0] for (index_, element) in enumerate(data_points)}
		# print("clusters: ", clusters)
		# print("cluster_mapping: ", cluster_mapping)
		# print("cluster_mapping key: ", set(cluster_mapping.keys()))
		# print("cluster_mapping_save: ", cluster_mapping_save)
		# print("cluster_mapping_save key: ", set(cluster_mapping_save.keys()), "\n")

		# Loop until only one cluster remains
		while (set(cluster_mapping.keys()) != set(cluster_mapping_save.keys())):

			cluster_mapping_save = cluster_mapping.copy()
			# print("cluster_mapping: ", cluster_mapping)
			# print("cluster_mapping_save: ", cluster_mapping_save, "\n")

			# Find the two closest clusters
			min_distance = np.inf
			min_i, min_j = -1, -1

			for i in range(len(clusters)):

				if len(clusters[i]) > 1:

					cluster_center_i = np.mean(clusters[i], axis = 0)
				else:

					cluster_center_i = clusters[i][0]

				for j in range(i + 1, len(clusters)):

					if len(clusters[j]) > 1:

						cluster_center_j = np.mean(clusters[j], axis = 0)
					else:
						cluster_center_j = clusters[j][0]

					dist = np.linalg.norm(cluster_center_i - cluster_center_j)
					# print("dist: ", dist)

					if dist < min_distance and dist < threshold:

						min_distance = dist
						min_i, min_j = i, j

			if min_i != -1 and min_j != -1:

				# print("min_i, min_j: ", min_i, min_j)
			
				# Merge the two closest clusters
				clusters[min_i] += clusters[min_j]
				del clusters[min_j]

				cluster_mapping[min_i] = np.append(cluster_mapping[min_i], cluster_mapping[min_j])
				del cluster_mapping[min_j]

				# Print cluster assignments
				i = 0
				cluster = {}
				for cluster_id, points in cluster_mapping.items():

					cluster[i] = np.array(points)
					i += 1

				cluster_mapping = cluster
			else:

		# 		print("min_i, min_j: ", min_i, min_j)
				pass

			# print("clusters: ", clusters)
			# print("cluster_mapping: ", cluster_mapping)
			# print("cluster_mapping key: ", set(cluster_mapping.keys()))
			# print("cluster_mapping_save: ", cluster_mapping_save)
			# print("cluster_mapping_save key: ", set(cluster_mapping_save.keys()), "\n")
			# print(set(cluster_mapping.keys()) == set(cluster_mapping_save.keys()))

		# print("clusters: ", clusters)
		# print("cluster_mapping: ", cluster_mapping)

		for key in cluster_mapping:

			cluster_mapping[key] = np.array(sorted(cluster_mapping[key]))

		# print("cluster_mapping: ", cluster_mapping)

		return cluster_mapping

	def Hungarian(self, Nc, cluster_center, cluster_set, Re):

		alpha = 0.0
		points = []
		for i in range(self.n):

			points.append(alpha*self.agents_state[i]["position"][0:2] + (1-alpha)*self.Ps[i])

		agents_len = len(points)

		for target in cluster_center:

			points.append(target)

		points = np.array(points)
		# print("points: " + str(points))

		points_len = len(points)
		targets_len = (points_len - agents_len)
		# print("agents_len: ", agents_len)
		# print("targets_len: ", targets_len)

		row_ind, col_ind = None, None
		if targets_len > agents_len or targets_len == agents_len:

			distances = distance.cdist(points, points)
			# print("distances: " + str(distances) + "\n")

			# Hungarian Algorithm
			cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
			cost_matrix = np.array(cost_matrix)
			# print("cost_matrix: ", cost_matrix)

			# print("Re: ", Re)
			# Define the vector
			vector = np.array(Re)

			# Compute the reciprocal of the vector
			reciprocal_vector = 1 / vector

			# Multiply each row of the matrix by the reciprocal vector using broadcasting
			cost_matrix = cost_matrix * reciprocal_vector
			# print("cost_matrix: ", cost_matrix)
			
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			# print("col_ind: ", col_ind)
		elif targets_len < agents_len:

			# print("points_len: ", points_len)
			# print("targets_len: ", targets_len)

			distances = distance.cdist(points, points)
			# print("points: " + str(points) + "\n")
			# print("distances: " + str(distances) + "\n")

			cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
			cost_matrix = np.array(cost_matrix)
			# print("cost_matrix: ", cost_matrix)

			for i in range(len(gain)):

				if gain[i] > 1:

					cost_matrix[:,i] *= 1/gain[i]

			# flip_ = np.inf*np.ones(agents_len - (points_len - agents_len))
			# hold = []
			# # print("flip_: ", flip_)

			# for i in range(agents_len):

			# 	if i >= points_len - agents_len:

			# 		hold.append(np.hstack((flip_, cost_matrix[i])))
			# 	else:
			# 		hold.append(np.hstack((cost_matrix[i], flip_)))

			# cost_matrix = np.array(hold)
			# print("cost_matrix: ", cost_matrix)

			# hold = np.inf*np.ones([agents_len, agents_len*targets_len])

			# for i in range(agents_len):

			# 	hold[i,i*targets_len:(i+1)*targets_len] = cost_matrix[i,:]

			# print("hold: ", hold)
			# cost_matrix = hold

			# row_ind, col_ind = linear_sum_assignment(cost_matrix)
			# col_ind = col_ind%targets_len

			row_ind, col_ind = linear_sum_assignment(cost_matrix)

			col_sol = {str(i): [] for i in range(agents_len)}

			for (row, col) in zip(row_ind, col_ind):

				col_sol[str(row)] = col

			# List with one missing number
			sequence_num = list(range(0, agents_len))
			missing_numbers = [num for num in sequence_num if num not in row_ind]
			missing_numbers_hold = missing_numbers
			# print("missing_number: ", missing_numbers)

			# Only for excessive assignment
			# points_ = []
			# for i_ in missing_numbers:

			# 	points_.append(cameras[i_].sweet_spot)

			# agents_len_ = len(points_)

			# for position_ in targets_position:

			# 	points_.append(position_)

			# points_ = np.array(points_)
			# points_len_ = len(points_)

			# distances_ = distance.cdist(points_, points_)
			# cost_matrix = [row[agents_len_:points_len_] for (row, i) in zip(distances_, range(len(distances_))) if i < agents_len_]
			# cost_matrix_missing = np.array(cost_matrix)

			# row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
			# # print("row_ind_missing: ", row_ind_missing)
			# # print("col_ind_missing: ", col_ind_missing)

			# for (row, col) in zip(missing_numbers, col_ind_missing):

			# 	col_sol[str(row)] = col-100

			# print("col_sol: ", col_sol)
			# -------------------------------------------------------------------- #

			while len(missing_numbers) != 0:

				cost_matrix_missing = np.array(cost_matrix[missing_numbers])
				# print("cost_matrix_missing: ", cost_matrix_missing)

				row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
				# print("row_ind_missing: ", row_ind_missing)
				# print("col_ind_missing: ", col_ind_missing)

				for (row, col) in zip(missing_numbers, col_ind_missing):

					col_sol[str(row)] = col
					missing_numbers_hold.remove(row)

				# print("col_sol: ", col_sol)
				missing_numbers = missing_numbers_hold

			col_ind = np.zeros(agents_len)
			for key_, value_ in col_sol.items():

				col_ind[int(key_)] = value_

			# col_indices = np.hstack((col_indices, col_indices_missing))
			# print("col_ind: ", col_ind)

		# print("col_ind: ", col_ind)
		# print("agents_len: ", agents_len)
		# print("points_len: ", points_len)
		# print("points_len - agents_len: ", points_len - agents_len)

		# for i in range(len(col_ind)):

		# 	if col_ind[i] < (points_len - agents_len)-1:

		# 		col_ind[i] = col_ind[i]
		# 	elif col_ind[i] > (points_len - agents_len)-1:

		# 		col_ind[i] = col_ind[i] - (agents_len - (points_len - agents_len))

		print("col_ind: ", col_ind)

		return col_ind

	def BILP(self, Nc, cluster_center, cluster_set, Re):

		alpha = 0.0
		points = []
		for i in range(self.n):

			points.append(alpha*self.agents_state[i]["position"][0:2] + (1-alpha)*self.Ps[i])

		# points.append(alpha*self.P0[0:2] + (1-alpha)*self.P0s)
		# points.append(alpha*self.P1[0:2] + (1-alpha)*self.P1s)
		# points.append(alpha*self.P2[0:2] + (1-alpha)*self.P2s)
		# points.append(alpha*self.P3[0:2] + (1-alpha)*self.P3s)
		# points.append(alpha*self.P4[0:2] + (1-alpha)*self.P4s)

		agents_len = len(points)

		for target in cluster_center:

			points.append(target)

		points = np.array(points)
		# print("points: " + str(points))

		points_len = len(points)

		distances = distance.cdist(points, points)
		# print("distances: " + str(distances) + "\n")

		# Distance Matrix
		cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		cost_matrix = np.array(cost_matrix)
		# print("cost_matrix: ", cost_matrix)

		row, col = np.shape(cost_matrix)
		C = cost_matrix.reshape((1, row*col))[0]
		# print("C: ", C)

		A, B = None, None
		if (self.n == self.m):

			# A Matrix Formulation
			A = np.zeros(row*col)
			for i in range(row+col):

				if i == 0:

					for j in range(col):

						A[j] = -1

				if i < row and i > 0:

					temp = np.zeros(row*col)

					for j in range(col):

						temp[i*col+j] = -1
					
					A = np.vstack((A, temp))

				if i >= row:

					temp = np.zeros(row*col)
					temp[i-row:row*col:col] = -1
					
					A = np.vstack((A, temp))
			# print("A: ", A)

			# B Formulaton
			B = np.zeros(row+col)
			for i in range(row+col):

				if i < row:

					B[i] = -1
				elif i >= row:

					B[i] = -Re[i-row]
			# print("B: ", B)

		elif (self.n < self.m) and (self.n > Nc):

			# A Matrix Formulation
			A = np.zeros(row*col)
			for i in range(row):

				if i == 0:

					for j in range(col):

						A[j] = -1

				if i < row and i > 0:

					temp = np.zeros(row*col)

					for j in range(col):

						temp[i*col+j] = -1
					
					A = np.vstack((A, temp))
			# print("A: ", A)

			# B Formulaton
			B = np.zeros(row)
			for i in range(row):

				if i < row:

					B[i] = -1
			# print("B: ", B)

		# Bounds for each variable (x0 and x1) as binary (0 or 1)
		bounds = [(0, 1)] * len(C)
		bounds = tuple(bounds)
		# print("bounds: ", bounds)

		# Solve the binary linear programming problem
		res = linprog(C, A_ub=A, b_ub=B, bounds=bounds, method='highs')

		# print("Optimal solution:")
		# print(f"x: {res.x}")
		# print("Optimal value:", res.fun)

		# col_ind
		col_ind = np.zeros(row)
		for i in range(row):

			for j in range(col):

				if res.x[i*col+j] > 0:
					
					col_ind[i] = j
		# print("col_ind: ", col_ind)

		return col_ind

	def Allocation(self):

		# Agglomerative Hierarchical Clustering
		# targets_position = np.array([self.P5[0:2], self.P6[0:2], self.P7[0:2], self.P8[0:2], self.P9[0:2]])
		targets_position = np.array([state["position"][0:2] for state in self.targets_state])
		# print("targets_position: ", targets_position)

		GCM = np.mean(targets_position, axis = 0)
		print("GCM: ", GCM)
		
		cluster_set = self.Agglomerative_Hierarchical_Clustering()
		print("cluster_set: ", cluster_set)

		cluster_center, Re = [], []
		for key, value in cluster_set.items():

			if len(value) > 1:

				# print("targets_position[value]: ", targets_position[value])
				# print(np.mean(targets_position[value], axis = 0))

				cluster_center.append(np.mean(targets_position[value], axis = 0))
			else:

				cluster_center.append(targets_position[value][0])

			Re.append(len(value))
		# print("Re: ", Re)

		cluster_center = np.array(cluster_center)
		print("cluster_center: ", cluster_center)

		Nc = len(cluster_set)
		print("Number of agent: ", self.n)
		print("Number of target: ", self.m)
		print("Number of cluster: ", Nc)

		col_ind = None
		if (self.n == self.m):

			col_ind = self.BILP(Nc, cluster_center, cluster_set, Re)
		elif (self.n < self.m) and (self.n < Nc):

			col_ind = self.Hungarian(Nc, cluster_center, cluster_set, Re)
		elif (self.n < self.m) and (self.n == Nc):

			col_ind = self.Hungarian(Nc, cluster_center, cluster_set, Re)
		elif (self.n < self.m) and (self.n > Nc):

			col_ind = self.BILP(Nc, cluster_center, cluster_set, Re)

		''' ---------- Herding Algorithm ---------- '''
		Pd = []; ra = 1;
		# for key, value in cluster_set.items():

		# 	df = (cluster_center[key] - GCM)
		# 	Af = GCM + df + (df/np.linalg.norm(df))*ra*np.sqrt(len(value))

		# 	if len(cluster_set) > 1:

		# 		Pd.append(Af)
		# 	else:

		# 		Af = np.zeros(2)
		# 		Pd.append(Af)

		dg_p = {key: [] for key in range(len(cluster_set.keys()))}
		for key, value in cluster_set.items():

			m = len(value); j = 1
			delta = 80*(np.pi/180)
			df = (GCM - cluster_center[key]); df_n = df/np.linalg.norm(df)
			angle = np.arctan2(df_n[1], df_n[0])
			r = ra*np.sqrt(len(value))
			# r = 4*np.cos(20*(np.pi/180))

			if angle < 0:

				angle += 2 * np.pi

			for i in range(m):

				if m == 1:

					j = 0
				else:

					j = ( (2*(i+1)-m-1)/(2*m-2) )

				theta = delta*j + np.pi + angle
				d = cluster_center[key] + np.array([r*np.cos(theta),\
													r*np.sin(theta)])
				dg_p[key].append(d.tolist())

			# print("m: ", m)
			# print("df: ", df)
			# print("Af: ", Af)
			# print("angle: ", angle)
			# print("dg_p: ", dg_p)
		# print("dg_p: ", dg_p)

		# self.Pd = Pd
		self.dg_p = dg_p
		self.cluster_set = cluster_set
		self.col_ind = col_ind

	'''
	def One_hop_neighbor_Team_One(self):

		points = np.array([self.P0[0:2], self.P1[0:2], self.P2[0:2], self.P3[0:2], self.P4[0:2]])

		if len(points) > 2:

			# Generate random points
			points = points

			# Create Delaunay Triangulation
			tri = Delaunay(points)

			# Find one-hop neighbors for each point
			one_hop_neighbors = [[] for _ in range(len(points))]

			# print("one_hop_neighbors: ", one_hop_neighbors)

			for simplex in tri.simplices:

				# print("simplex: ", simplex)

				for point_index in simplex:

					for neighbor_index in simplex:

						if point_index != neighbor_index and neighbor_index not in one_hop_neighbors[point_index]:

							one_hop_neighbors[point_index].append(neighbor_index)

		elif len(points) == 2:

			one_hop_neighbors = [[1], [0]]
		else:

			one_hop_neighbors = [[1], [0]]

		self.one_hop_neighbors = one_hop_neighbors
		# print("one_hop_neighbors: ", one_hop_neighbors)

		# return one_hop_neighbors
	'''
	'''
	def One_hop_neighbor_Team_Multiple(self):

		n = len(self.cluster_set)
		points = np.array([self.P0[0:2], self.P1[0:2], self.P2[0:2], self.P3[0:2], self.P4[0:2]])

		# Find one-hop neighbors for each point
		one_hop_neighbors = [[] for _ in range(len(self.col_ind))]
		# print("one_hop_neighbors: ", one_hop_neighbors)

		for j in range(n):

			agents = []
			agents_id = []

			for (i, index) in zip(self.col_ind, range(len(self.col_ind))):

				if i == j:

					agents.append(points[index])
					agents_id.append(index)

			# print("agents: ", agents)
			# print("agents_id: ", agents_id)

			if len(agents) > 2:

				# Create Delaunay Triangulation
				tri = Delaunay(agents)

				# Find one-hop neighbors for each point
				one_hop_neighbors_temp = [[] for _ in range(len(agents))]

				# print("one_hop_neighbors: ", one_hop_neighbors)

				for simplex in tri.simplices:

					# print("simplex: ", simplex)

					for point_index in simplex:

						for neighbor_index in simplex:

							if point_index != neighbor_index and neighbor_index not in one_hop_neighbors_temp[point_index]:

								one_hop_neighbors_temp[point_index].append(neighbor_index)

				for (index, i) in zip(agents_id, range(len(agents_id))):

					one_hop_neighbors[index] = np.array(agents_id)[one_hop_neighbors_temp[i]].tolist()

				# print("one_hop_neighbors: ", one_hop_neighbors)
			elif len(agents) == 2:

					# print("agents_id[0]: ", agents_id[0])
					# print("agents_id[1]: ", agents_id[1])

					one_hop_neighbors[agents_id[0]].append(agents_id[1])
					one_hop_neighbors[agents_id[1]].append(agents_id[0])

					# print("one_hop_neighbors agents_id[0]: ", one_hop_neighbors[agents_id[0]])
					# print("one_hop_neighbors agents_id[1]: ", one_hop_neighbors[agents_id[1]])
			else:

				one_hop_neighbors[agents_id[0]].append(agents_id[0])

			# print("one_hop_neighbors: ", one_hop_neighbors)
		# print("one_hop_neighbors: ", one_hop_neighbors)

		self.one_hop_neighbors = one_hop_neighbors
		# return one_hop_neighbors
	'''

	def One_Hop_Neighbor(self):

		# points = np.array([self.P0[0:2], self.P1[0:2], self.P2[0:2], self.P3[0:2], self.P4[0:2]])
		points = np.array([state["position"][0:2] for state in self.agents_state])

		# Team Formulation
		Team = [[i] for i in range(len(self.col_ind))]

		for (index, value) in enumerate(self.col_ind):

			for (index_, value_) in enumerate(self.col_ind):

				if (index_ != index) and (value_ == value):

					Team[index].append(index_)

		print("Team: ", Team)

		# Find one-hop neighbors for each point
		one_hop_neighbors = [[] for _ in range(len(self.col_ind))]
		# print("one_hop_neighbors: ", one_hop_neighbors)

		for (key, value_) in self.cluster_set.items():

			agents = []
			agents_id = []

			for (index, value) in enumerate(self.col_ind):

				if value == key:

					agents.append(points[index])
					agents_id.append(index)
			# print("agents: ", agents)
			# print("agents_id: ", agents_id)

			if len(agents) > 2:

				# Create Delaunay Triangulation
				tri = Delaunay(agents)

				# Find one-hop neighbors for each point
				one_hop_neighbors_temp = [[] for _ in range(len(agents))]

				for simplex in tri.simplices:

					# print("simplex: ", simplex)

					for point_index in simplex:

						for neighbor_index in simplex:

							if point_index != neighbor_index and neighbor_index not in one_hop_neighbors_temp[point_index]:

								one_hop_neighbors_temp[point_index].append(neighbor_index)

				for (index, value) in enumerate(agents_id):

					one_hop_neighbors[value] = np.array(agents_id)[one_hop_neighbors_temp[index]].tolist()

				# print("one_hop_neighbors: ", one_hop_neighbors)
			elif len(agents) == 2:

					# print("agents_id[0]: ", agents_id[0])
					# print("agents_id[1]: ", agents_id[1])

					one_hop_neighbors[agents_id[0]].append(agents_id[1])
					one_hop_neighbors[agents_id[1]].append(agents_id[0])

					# print("one_hop_neighbors agents_id[0]: ", one_hop_neighbors[agents_id[0]])
					# print("one_hop_neighbors agents_id[1]: ", one_hop_neighbors[agents_id[1]])
			elif len(agents) == 1:

				if len(points) > 2:

					# Create Delaunay Triangulation
					tri = Delaunay(points)

					for simplex in tri.simplices:

						# print("simplex: ", simplex)

						for point_index in simplex:

							if point_index == agents_id[0]:

								for neighbor_index in simplex:

									if point_index != neighbor_index and neighbor_index not in one_hop_neighbors[point_index]:

										one_hop_neighbors[point_index].append(neighbor_index)
				elif len(points) == 2:

					one_hop_neighbors = [[1], [0]]
				else:

					one_hop_neighbors = [[0]]

		self.Team = Team
		self.one_hop_neighbors = [sorted(sublist) for sublist in one_hop_neighbors]
		print("one_hop_neighbors: ", self.one_hop_neighbors)


		'''--------- Herding Algorithm ----------'''
		# Find unique elements in the list
		unique_elements = set(self.col_ind)

		# Create a dictionary to hold the indices of each unique element
		indices_dict = {int(elem): [] for elem in unique_elements}

		# Populate the dictionary with indices
		for index, value in enumerate(self.col_ind):

			indices_dict[value].append(index)

		# Print the dictionary
		# print("indices_dict: ", indices_dict)

		Pd = {index: None for index in range(len(points))}
		for key, value in indices_dict.items():

			dog = []
			dog.extend(points[value])
			dog_len = len(dog)

			for dp in self.dg_p[key]:

				dog.append(dp)

			# print("dog: ", dog)
			dog = np.array(dog)
			points_len = len(dog)

			distances = distance.cdist(dog, dog)
			cost_matrix = [row[dog_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < dog_len]
			cost_matrix = np.array(cost_matrix)

			row_ind_dp, col_ind_dp = linear_sum_assignment(cost_matrix)
			# print("row_ind_dp: ", row_ind_dp)
			# print("col_ind_dp: ", col_ind_dp)

			for row, col in zip(row_ind_dp, col_ind_dp):

				Pd[value[row]] = self.dg_p[key][col]

		# print("Pd: ", Pd)
		self.Pd = Pd

	'''
	def K_means(self, targets, cameras):

		points = []
		targets_position = np.array([target[0] for target in targets])

		alpha = 0.0
		for camera in cameras:

			points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

		agents_len = len(points)
		
		for target_position in targets_position:

			points.append(target_position)

		points = np.array(points)

		# print("points: " + str(points))

		points_len = len(points)

		distances = distance.cdist(points, points)
		# print("points: " + str(points) + "\n")
		# print("distances: " + str(distances) + "\n")

		# Hungarian Algorithm
		# cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		# cost_matrix = np.array(cost_matrix)
		# row_ind, col_ind = linear_sum_assignment(cost_matrix)

		# Step 2 - K-Means Algorithm to update Cluster member and its members
		# cluster_centers = np.array([targets_position[element] for element in col_ind])
		# cluster_centers = points[0:4]
		cluster_centers = points[0:agents_len]
		# print("cluster_centers: " + str(cluster_centers))
		data = targets_position
		# print("data: " + str(data))
		alpha = 0.3

		for i in range(100):

			# Step 2: Assignment Step - Assign each data point to the nearest centroid
			cluster_labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2), axis=1)

			# Step 3: Update Step - Recalculate the centroids
			mean = np.array([data[cluster_labels == i].mean(axis=0) for i in range(agents_len)])
			# print("mean: " + str(mean))
			# print("cluster_labels: " + str(cluster_labels))

			for j in range(agents_len):

				count = 0

				if (np.isnan(mean[j]).any()):

					cluster_centers[j] = cameras[j].sweet_spot
				else:

					new_centroids = (1 - alpha)*cluster_centers[j] + alpha*mean[j]
					# new_centroids = np.array([data[cluster_labels == j].mean(axis = 0)])

					# print("new_centroids: " + str(new_centroids))

					# Check for convergence
					if np.allclose(cluster_centers, new_centroids):

						count += 1
					else:

						cluster_centers[j] = new_centroids

				if count == agents_len:

					break

		for (i, item_) in enumerate(cluster_centers):

			if (np.isnan(item_).any()):

				print(True)

				cluster_centers[i] = cameras[i].sweet_spot

		# print("labels: " + str(cluster_labels))
		# print("Centroid: " + str(cluster_centers))

		return cluster_labels
	'''
	'''
	def FuzzyCMeans(self, targets, cameras):

		points = []
		targets_position = np.array([target[0] for target in targets])

		alpha = 0.0
		for camera in cameras:

			points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

		c = len(points)

		# Fuzzy C-Means algorithm
		cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(targets_position.T, c, 2.0, error = 0.001, maxiter = 1000, seed = 42)
		cluster_centers = cntr
		cluster_membership = np.argmax(u, axis = 0)  # Get the cluster membership for each point

		return cluster_centers, cluster_membership
	'''

	def Targets_Motion(self, time, start, interval):

		print("time: ", time)

		speed_gain = 0.50

		seed_key = {1: "103", 2: "106", 3: "143", 4: "279", 5: "351", 6: "333", 7: "555", 8: "913", 9: "3821", 10: "5214",
					11: "5232", 12: "5246", 13: "5532"}

		# positions = np.array([self.P5[0:2], self.P6[0:2], self.P7[0:2], self.P8[0:2], self.P9[0:2]])
		positions = np.array([state["position"][0:2] for state in self.targets_state])
		# print("positions: ", positions)

		# if int(time) >= 10 and int(time) <= 70:
		if int(time) >= start and int(time) <= start+interval:

			# Simulation parameters
			time_step = 0.1  # Time step in seconds
			min_distance = 1.0  # Minimum distance between points to avoid collision
			boundary_margin = 3  # Minimum distance from the boundary
			tracker_margin = 1  # Minimum distance from the boundary

			# Initialize point positions and velocities
			# positions = [self.P5[0:2], self.P6[0:2], self.P7[0:2], self.P8[0:2], self.P9[0:2]]
			positions = np.array([state["position"][0:2] for state in self.targets_state])
			# positions += velocities * time_step
			# print("positions: " + str(positions))

			# Change direction of velocities every 100 timestep
			# if int(time)%10 == 0 and int(time) <= 60:
			if int(time)%10 == 0 and int(time) <= interval:
				
				np.random.seed(int(seed_key[int(time/10)]))
				self.velocities = (float(speed_gain)/0.5)*(np.random.rand(len(positions), 2) - 0.5)*0.7

				# v_ = np.array([(0,0), (1,0), (0,0), (0,0), (0,0)])
				# self.velocities = (float(speed_gain)/0.5)*(v_)*0.2
 
				# v_ = np.array([(0,0), (0,0), (-1,-1), (0,1), (1,0)])
				# self.velocities = (float(speed_gain)/0.5)*(v_)*0.2

				# print("seed: " + str(int(seed_key[int(run_step/100)])))
				# print("velocities: " + str(velocities))

			# Check for collisions and adjust velocities if necessary
			for i in range(len(positions)):

				# for j in range(i + 1, 4):
				for j in range(len(positions)):

					if j != i:
					
						dist = np.linalg.norm(positions[i] - positions[j])

						if dist < min_distance:

							# Adjust velocities to avoid collision
							direction = positions[i] - positions[j]
							self.velocities[i] = +(direction/np.linalg.norm(direction))*0.5
							self.velocities[j] = -(direction/np.linalg.norm(direction))*0.5

				if abs(positions[i][0] - 0) <= boundary_margin or abs(positions[i][0] - 25) <= boundary_margin:

					self.velocities[i][0] *= -0.5  # Reverse x-direction velocity
				if abs(positions[i][1] - 0) <= boundary_margin or abs(positions[i][1] - 25) <= boundary_margin:

					self.velocities[i][1] *= -0.5  # Reverse y-direction velocity
		# elif int(time) < 10 or int(time) > 70:
		elif int(time) < start or int(time) > interval:

			# self.velocities = np.array([(1e-6, 1e-6), (1e-6, 1e-6), (1e-6, 1e-6), (1e-6, 1e-6), (1e-6, 1e-6)])
			self.velocities = np.array([(1e-6, 1e-6) for _ in range(self.m)])

		# print("Target Velocity: ", self.velocities)

		Target_cmd_vel = TwistStamped()

		# Target_cmd_vel.twist.linear.x = 2.0*self.velocities[0][0]
		# Target_cmd_vel.twist.linear.y = 2.0*self.velocities[0][1]
		# Target_cmd_vel.twist.linear.z = 9.8 - self.P5[2]
		# self.UAV5_cmd_vel.publish(Target_cmd_vel)
		# self.UAV5_hold_vel.publish(Target_cmd_vel)

		# Target_cmd_vel.twist.linear.x = 2.0*self.velocities[1][0]
		# Target_cmd_vel.twist.linear.y = 2.0*self.velocities[1][1]
		# Target_cmd_vel.twist.linear.z = 9.8 - self.P6[2]
		# self.UAV6_cmd_vel.publish(Target_cmd_vel)
		# self.UAV6_hold_vel.publish(Target_cmd_vel)

		# Target_cmd_vel.twist.linear.x = 2.0*self.velocities[2][0]
		# Target_cmd_vel.twist.linear.y = 2.0*self.velocities[2][1]
		# Target_cmd_vel.twist.linear.z = 9.8 - self.P7[2]
		# self.UAV7_cmd_vel.publish(Target_cmd_vel)
		# self.UAV7_hold_vel.publish(Target_cmd_vel)

		# Target_cmd_vel.twist.linear.x = 2.0*self.velocities[3][0]
		# Target_cmd_vel.twist.linear.y = 2.0*self.velocities[3][1]
		# Target_cmd_vel.twist.linear.z = 9.8 - self.P8[2]
		# self.UAV8_cmd_vel.publish(Target_cmd_vel)
		# self.UAV8_hold_vel.publish(Target_cmd_vel)

		# Target_cmd_vel.twist.linear.x = 2.0*self.velocities[4][0]
		# Target_cmd_vel.twist.linear.y = 2.0*self.velocities[4][1]
		# Target_cmd_vel.twist.linear.z = 9.8 - self.P9[2]
		# self.UAV9_cmd_vel.publish(Target_cmd_vel)
		# self.UAV9_hold_vel.publish(Target_cmd_vel)

		for i in range(self.m):

			Target_cmd_vel.twist.linear.x = 2.0*self.velocities[i][0]
			Target_cmd_vel.twist.linear.y = 2.0*self.velocities[i][1]
			Target_cmd_vel.twist.linear.z = 9.8 - self.targets_state[i]["position"][2]
			self.target_cmd_vel[i].publish(Target_cmd_vel)
			self.target_hold_vel[i].publish(Target_cmd_vel)

	def Transmission(self):

		#----- State Information -----@

		# # UAV0
		# uav0_pose = Pose()
		# uav0_twist = Twist()
		# uav0_pose.position.x = self.P0[0]; uav0_pose.position.y = self.P0[1]; uav0_pose.position.z = self.P0[2]
		# uav0_pose.orientation.x = self.P0o[0]; uav0_pose.orientation.y = self.P0o[1]; uav0_pose.orientation.z = self.P0o[2];
		# uav0_pose.orientation.w = self.P0o[3]
		# uav0_twist.linear.x = 0.0; uav0_twist.linear.y = 0.0; uav0_twist.linear.z = 0.0

		# # UAV1
		# uav1_pose = Pose()
		# uav1_twist = Twist()
		# uav1_pose.position.x = self.P1[0]; uav1_pose.position.y = self.P1[1]; uav1_pose.position.z = self.P1[2]
		# uav1_pose.orientation.x = self.P1o[0]; uav1_pose.orientation.y = self.P1o[1]; uav1_pose.orientation.z = self.P1o[2];
		# uav1_pose.orientation.w = self.P1o[3]
		# uav1_twist.linear.x = 0.0; uav1_twist.linear.y = 0.0; uav1_twist.linear.z = 0.0

		# # UAV2
		# uav2_pose = Pose()
		# uav2_twist = Twist()
		# uav2_pose.position.x = self.P2[0]; uav2_pose.position.y = self.P2[1]; uav2_pose.position.z = self.P2[2]
		# uav2_pose.orientation.x = self.P2o[0]; uav2_pose.orientation.y = self.P2o[1]; uav2_pose.orientation.z = self.P2o[2];
		# uav2_pose.orientation.w = self.P2o[3]
		# uav2_twist.linear.x = 0.0; uav2_twist.linear.y = 0.0; uav2_twist.linear.z = 0.0

		# # UAV3
		# uav3_pose = Pose()
		# uav3_twist = Twist()
		# uav3_pose.position.x = self.P3[0]; uav3_pose.position.y = self.P3[1]; uav3_pose.position.z = self.P3[2]
		# uav3_pose.orientation.x = self.P3o[0]; uav3_pose.orientation.y = self.P3o[1]; uav3_pose.orientation.z = self.P3o[2];
		# uav3_pose.orientation.w = self.P3o[3]
		# uav3_twist.linear.x = 0.0; uav3_twist.linear.y = 0.0; uav3_twist.linear.z = 0.0

		# # UAV4
		# uav4_pose = Pose()
		# uav4_twist = Twist()
		# uav4_pose.position.x = self.P4[0]; uav4_pose.position.y = self.P4[1]; uav4_pose.position.z = self.P4[2]
		# uav4_pose.orientation.x = self.P4o[0]; uav4_pose.orientation.y = self.P4o[1]; uav4_pose.orientation.z = self.P4o[2];
		# uav4_pose.orientation.w = self.P4o[3]
		# uav4_twist.linear.x = 0.0; uav4_twist.linear.y = 0.0; uav4_twist.linear.z = 0.0


		# # UAV5
		# uav5_pose = Pose()
		# uav5_twist = Twist()
		# uav5_pose.position.x = self.P5[0]; uav5_pose.position.y = self.P5[1]; uav5_pose.position.z = self.P5[2]
		# uav5_twist.linear.x = self.P5v[0]; uav5_twist.linear.y = self.P5v[1]; uav5_twist.linear.z = self.P5v[2]

		# # UAV6
		# uav6_pose = Pose()
		# uav6_twist = Twist()
		# uav6_pose.position.x = self.P6[0]; uav6_pose.position.y = self.P6[1]; uav6_pose.position.z = self.P6[2]
		# uav6_twist.linear.x = self.P6v[0]; uav6_twist.linear.y = self.P6v[1]; uav6_twist.linear.z = self.P6v[2]

		# # UAV7
		# uav7_pose = Pose()
		# uav7_twist = Twist()
		# uav7_pose.position.x = self.P7[0]; uav7_pose.position.y = self.P7[1]; uav7_pose.position.z = self.P7[2]
		# uav7_twist.linear.x = self.P7v[0]; uav7_twist.linear.y = self.P7v[1]; uav7_twist.linear.z = self.P7v[2]

		# # UAV8
		# uav8_pose = Pose()
		# uav8_twist = Twist()
		# uav8_pose.position.x = self.P8[0]; uav8_pose.position.y = self.P8[1]; uav8_pose.position.z = self.P8[2]
		# uav8_twist.linear.x = self.P8v[0]; uav8_twist.linear.y = self.P8v[1]; uav8_twist.linear.z = self.P8v[2]

		# # UAV9
		# uav9_pose = Pose()
		# uav9_twist = Twist()
		# uav9_pose.position.x = self.P9[0]; uav9_pose.position.y = self.P9[1]; uav9_pose.position.z = self.P9[2]
		# uav9_twist.linear.x = self.P9v[0]; uav9_twist.linear.y = self.P9v[1]; uav9_twist.linear.z = self.P9v[2]

		output_pose, output_twist = [], []
		output_name = np.concatenate((self.agents_name, self.targets_name)).tolist()
		# print("output_name: ", output_name)

		for i in range(self.n+self.m):

			uav_pose = Pose()
			uav_twist = Twist()

			if i < self.n:

				uav_pose.position.x = self.agents_state[i]["position"][0]
				uav_pose.position.y = self.agents_state[i]["position"][1]
				uav_pose.position.z = self.agents_state[i]["position"][2]
				uav_pose.orientation.x = self.agents_state[i]["orientation"][0]
				uav_pose.orientation.y = self.agents_state[i]["orientation"][1]
				uav_pose.orientation.z = self.agents_state[i]["orientation"][2]
				uav_pose.orientation.w = self.agents_state[i]["orientation"][3]

				uav_twist.linear.x = 0.0; uav_twist.linear.y = 0.0; uav_twist.linear.z = 0.0
			else:

				j = i - self.n

				uav_pose.position.x = self.targets_state[j]["position"][0]
				uav_pose.position.y = self.targets_state[j]["position"][1]
				uav_pose.position.z = self.targets_state[j]["position"][2]
				uav_twist.linear.x = self.targets_state[j]["velocity"][0]
				uav_twist.linear.y = self.targets_state[j]["velocity"][1]
				uav_twist.linear.z = self.targets_state[j]["velocity"][2]

			output_pose.append(uav_pose)
			output_twist.append(uav_twist)

		output_ = ModelStates()
		# output_.name = ["uav0", "uav1", "uav2", "uav3", "uav4", "uav5", "uav6", "uav7", "uav8", "uav9"]
		# output_.pose = [uav0_pose, uav1_pose, uav2_pose, uav3_pose, uav4_pose,\
		# 				uav5_pose, uav6_pose, uav7_pose, uav8_pose, uav9_pose]
		# output_.twist = [uav0_twist, uav1_twist, uav2_twist, uav3_twist, uav4_twist,\
		# 				uav5_twist, uav6_twist, uav7_twist, uav8_twist, uav9_twist]
		output_.name = output_name
		output_.pose = output_pose
		output_.twist = output_twist

		self.UAV_StateI.publish(output_)

		# -------- Link Information -------@
		# cam0_pose = Pose()
		# cam0_pose.position.x = self.Cam0[0]; cam0_pose.position.y = self.Cam0[1]; cam0_pose.position.z = self.Cam0[2]
		# cam0_pose.orientation.x = self.Cam0o[0]; cam0_pose.orientation.y = self.Cam0o[1]; cam0_pose.orientation.z = self.Cam0o[2];
		# cam0_pose.orientation.w = self.Cam0o[3]

		# # UAV1
		# cam1_pose = Pose()
		# cam1_pose.position.x = self.Cam1[0]; cam1_pose.position.y = self.Cam1[1]; cam1_pose.position.z = self.Cam1[2]
		# cam1_pose.orientation.x = self.Cam1o[0]; cam1_pose.orientation.y = self.Cam1o[1]; cam1_pose.orientation.z = self.Cam1o[2];
		# cam1_pose.orientation.w = self.Cam1o[3]

		# # UAV2
		# cam2_pose = Pose()
		# cam2_pose.position.x = self.Cam2[0]; cam2_pose.position.y = self.Cam2[1]; cam2_pose.position.z = self.Cam2[2]
		# cam2_pose.orientation.x = self.Cam2o[0]; cam2_pose.orientation.y = self.Cam2o[1]; cam2_pose.orientation.z = self.Cam2o[2];
		# cam2_pose.orientation.w = self.Cam2o[3]

		# # UAV3
		# cam3_pose = Pose()
		# cam3_pose.position.x = self.Cam3[0]; cam3_pose.position.y = self.Cam3[1]; cam3_pose.position.z = self.Cam3[2]
		# cam3_pose.orientation.x = self.Cam3o[0]; cam3_pose.orientation.y = self.Cam3o[1]; cam3_pose.orientation.z = self.Cam3o[2];
		# cam3_pose.orientation.w = self.Cam3o[3]

		# # UAV4
		# cam4_pose = Pose()
		# cam4_pose.position.x = self.Cam4[0]; cam4_pose.position.y = self.Cam4[1]; cam4_pose.position.z = self.Cam4[2]
		# cam4_pose.orientation.x = self.Cam4o[0]; cam4_pose.orientation.y = self.Cam4o[1]; cam4_pose.orientation.z = self.Cam4o[2];
		# cam4_pose.orientation.w = self.Cam4o[3]

		output_pose = []
		for i in range(self.n):

			cam_pose = Pose()
			cam_pose.position.x = self.Cams_state[i]["position"][0]
			cam_pose.position.y = self.Cams_state[i]["position"][1]
			cam_pose.position.z = self.Cams_state[i]["position"][2]
			cam_pose.orientation.x = self.Cams_state[i]["orientation"][0]
			cam_pose.orientation.y = self.Cams_state[i]["orientation"][1]
			cam_pose.orientation.z = self.Cams_state[i]["orientation"][2]
			cam_pose.orientation.w = self.Cams_state[i]["orientation"][3]

			output_pose.append(cam_pose)

		output_ = LinkStates()
		# output_.name = ["uav0::cgo3_camera_link", "uav1::cgo3_camera_link", "uav2::cgo3_camera_link",\
		# 				"uav3::cgo3_camera_link", "uav4::cgo3_camera_link"]
		# output_.pose = [cam0_pose, cam1_pose, cam2_pose, cam3_pose, cam4_pose]
		output_.name = self.Cams_name
		output_.pose = output_pose

		self.UAV_LinkI.publish(output_)

		# Command Information
		h_dim = MultiArrayDimension(label = "height", size = 12, stride = 1*2*12)
		w_dim = MultiArrayDimension(label = "width",  size = 2, stride = 1*2)
		c_dim = MultiArrayDimension(label = "channel", size = 4, stride = 4)
		layout = MultiArrayLayout(dim = [h_dim, w_dim, c_dim], data_offset = 0)

		print("col_ind: ", self.col_ind)
		print("cluster_set: ", self.cluster_set)
		print("Pd: ", self.Pd)

		# -1e6 -> "col_ind", -2e6 -> "one_hop_neighbors", -3e6 -> "Pd", -4e6 -> "cluster_set", -5e6 -> "teammate"
		# UAV 0
		# uav_ref = []
		# uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[0]))
		# uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[0])
		# # uav_ref.append(-3e6); uav_ref.extend(self.Pd[int(self.col_ind[0])].tolist())
		# uav_ref.append(-3e6); uav_ref.extend(self.Pd[0])
		# uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[0]].tolist())
		# uav_ref.append(-5e6); uav_ref.extend(self.Team[0])
		# # print("uav_ref: ", uav_ref)

		# # uav0_ref = [(int(self.col_ind[0]))]
		# # uav0_ref.extend([self.one_hop_neighbors[0][0], self.one_hop_neighbors[0][1]])
		# # uav0_ref.extend(self.Pd[int(self.col_ind[0])].tolist());

		# # if len(self.cluster_set[self.col_ind[0]]) > 1:

		# # 	uav0_ref.extend(self.cluster_set[self.col_ind[0]].tolist())
		# # else:

		# # 	uav0_ref.extend(self.cluster_set[self.col_ind[0]])

		# output_ = Float64MultiArray(data = uav_ref, layout = layout)
		# self.UAV0_Ref.publish(output_)

		# # UAV 1
		# uav_ref = []
		# uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[1]))
		# uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[1])
		# uav_ref.append(-3e6); uav_ref.extend(self.Pd[1])
		# uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[1]].tolist())
		# uav_ref.append(-5e6); uav_ref.extend(self.Team[1])

		# output_ = Float64MultiArray(data = uav_ref, layout = layout)
		# self.UAV1_Ref.publish(output_)

		# # UAV 2
		# uav_ref = []
		# uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[2]))
		# uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[2])
		# uav_ref.append(-3e6); uav_ref.extend(self.Pd[2])
		# uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[2]].tolist())
		# uav_ref.append(-5e6); uav_ref.extend(self.Team[2])

		# output_ = Float64MultiArray(data = uav_ref, layout = layout)
		# self.UAV2_Ref.publish(output_)

		# # UAV 3
		# uav_ref = []
		# uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[3]))
		# uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[3])
		# uav_ref.append(-3e6); uav_ref.extend(self.Pd[3])
		# uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[3]].tolist())
		# uav_ref.append(-5e6); uav_ref.extend(self.Team[3])

		# output_ = Float64MultiArray(data = uav_ref, layout = layout)
		# self.UAV3_Ref.publish(output_)

		# # UAV 4
		# uav_ref = []
		# uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[4]))
		# uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[4])
		# uav_ref.append(-3e6); uav_ref.extend(self.Pd[4])
		# uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[4]].tolist())
		# uav_ref.append(-5e6); uav_ref.extend(self.Team[4])

		# output_ = Float64MultiArray(data = uav_ref, layout = layout)
		# self.UAV4_Ref.publish(output_)

		for i in range(self.n):

			uav_ref = []
			uav_ref.append(-1e6); uav_ref.append(int(self.col_ind[i]))
			uav_ref.append(-2e6); uav_ref.extend(self.one_hop_neighbors[i])
			uav_ref.append(-3e6); uav_ref.extend(self.Pd[i])
			uav_ref.append(-4e6); uav_ref.extend(self.cluster_set[self.col_ind[i]].tolist())
			uav_ref.append(-5e6); uav_ref.extend(self.Team[i])

			output_ = Float64MultiArray(data = uav_ref, layout = layout)
			self.agent_Ref[i].publish(output_)

if __name__ == "__main__":

	try:
		rospy.init_node('Ground_Station')
		rate = rospy.Rate(100)
		GS = QGC()

		# while (GS.P0 is None) or (GS.P1 is None) or (GS.P2 is None) or (GS.P3 is None) or (GS.P4 is None) or\
		# 		(GS.P5 is None) or (GS.P6 is None) or (GS.P7 is None) or (GS.P8 is None) or (GS.P9 is None):
		while (GS.agents_state is None) or (GS.agents_name is None) or (GS.targets_state is None) or (GS.targets_name is None) or\
				(GS.UAV_currH is None) or (GS.Cams_state is None) or (GS.Cams_name is None) or\
				(GS.n is None) or (GS.m is None):

			rate.sleep()

		last = time()

		# Target Related Publisher
		target_cmd_vel, target_hold_vel = [], []
		for i in range(GS.m):

			UAV_cmd_vel = rospy.Publisher("/uav" + str(i+GS.n) + "/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=100)
			UAV_hold_vel = rospy.Publisher("/uav" + str(i+GS.n) + "/cmd_vel", TwistStamped, queue_size=100)

			target_cmd_vel.append(UAV_cmd_vel)
			target_hold_vel.append(UAV_hold_vel)
		GS.target_cmd_vel = target_cmd_vel
		GS.target_hold_vel = target_hold_vel
		# print("target_cmd_vel: ", target_cmd_vel)
		# print("target_hold_vel: ", target_hold_vel)

		# Agent Related Publisher
		agent_Ref = []
		for i in range(GS.n):

			UAV_Ref = rospy.Publisher("/uav" + str(i) + "/Ref", Float64MultiArray, queue_size = 100)
			agent_Ref.append(UAV_Ref)
		GS.agent_Ref = agent_Ref
		# print("agent_Ref: ", agent_Ref)

		while not rospy.is_shutdown():

			past = time()

			GS.UAV_Sweet_Spot()

			GS.Allocation()
			GS.One_Hop_Neighbor()

			# GS.Targets_Motion(np.round(time() - last, 2), 20, 60)
			GS.Transmission()

			print("Execution Time: " + str(time() - past) + "\n")

			rate.sleep()

	except rospy.ROSInterruptException:

		pass