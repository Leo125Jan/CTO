#!/usr/bin/python3

import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import re
import sys
import csv
import copy
import rospy
import random
import numpy as np
from QBSC import PTZcon
from time import sleep, time
from sensor_msgs.msg import Imu
from pyquaternion import Quaternion
from geometry_msgs.msg import Twist
from px4_mavros import Px4Controller
from sensor_msgs.msg import CameraInfo
from Estimator_Self import DepthEstimator
from gazebo_msgs.msg import ModelStates, LinkStates
from coverage.msg import BoundingBox, BoundingBoxes
from std_msgs.msg import Int64, Float64, Float64MultiArray
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp

class UAV():

	def __init__(self, id_):

		self.id = int(id_)

		# Variable of Control
		self.agents_state = None
		self.agents_name = None
		self.targets_state = None
		self.targets_name = None
		self.Cams_state = None
		self.Cams_name = None
		self.n, self.m = None, None
		
		self.cmd_vel = Twist()
		self.px4 = Px4Controller("uav0")

		size = int(25/0.1)
		self.current_heading = 0.0
		self.targetName = "Drone"
		self.pixel_size = 2.9*1e-6
		self.image_width = 1280
		self.FL_1x = 1280.785 # 6.5mm/2.9um
		self.FL_Curr = self.FL_1x
		self.img0, self.pixel0, self.camerainfo0 = np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0,0.0]])

		self.col_ind = None
		self.neighbor_index = None
		self.last_neighbor_topics = None
		self.last_index = None
		self.Pd = None
		self.cluster_set = None
		self.teammate = None

		self.neighbors_residual = {}
		self.neighbors_vel = {}
		self.neighbors_swt = {}
		self.neighbors_winbid = {}
		self.neighbors_taskcomplete = {}
		self.neighbors_resetready = {}
		self.neighbors_resetcomplete = {}
		self.neighbors_counting = {}
		self.neighbors_ess = {}
		self.neighbors_det = {}

		self.topic_list = ["/targetResdiual", "/CurrVel", "/SweetSpot", "/WinBid", "/TaskComplete", "/ResetReady",\
							"/ResetComplete", "/Counting", "/Essential", "/Detection"]
		self.varb_list = [self.neighbors_residual, self.neighbors_vel, self.neighbors_swt, self.neighbors_winbid,\
						self.neighbors_taskcomplete, self.neighbors_resetready, self.neighbors_resetcomplete,\
						self.neighbors_counting, self.neighbors_ess, self.neighbors_det]
		self.init_list = [np.array([[None, None]]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), None, False, False, False, None, None, None]

		# Publisher & Subscriber
		self.States_sub = rospy.Subscriber('/uav/StateI', ModelStates, self.state_callback, queue_size = 20, buff_size = 52428800)
		self.Link_sub = rospy.Subscriber('/uav/LinkI', LinkStates, self.link_callback, queue_size = 20, buff_size = 52428800)
		self.Ref_sub = rospy.Subscriber('/uav0/Ref', Float64MultiArray, self.reference_callback, queue_size = 20, buff_size = 52428800)

		self.uav0_camerainfo_sub = rospy.Subscriber("/uav0/camera/camera/color/camera_info", CameraInfo, self.CameraInfo0_callback, queue_size = 100)
		self.uav0_recognition_sub = rospy.Subscriber("/uav0/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition0_callback, queue_size = 100)

		self.neighbors_residual_sub = {}
		self.neighbors_vel_sub = {}
		self.neighbors_swt_sub = {}
		self.neighbors_winbid_sub = {}
		self.neighbors_taskcomplete_sub = {}
		self.neighbors_resetready_sub = {}
		self.neighbors_resetcomplete_sub = {}
		self.neighbors_counting_sub = {}
		self.neighbors_ess_sub = {}
		self.neighbors_det_sub = {}

		self.sub_list = [self.neighbors_residual_sub, self.neighbors_vel_sub, self.neighbors_swt_sub, self.neighbors_winbid_sub,\
						self.neighbors_taskcomplete_sub, self.neighbors_resetready_sub, self.neighbors_resetcomplete_sub,\
						self.neighbors_counting_sub, self.neighbors_ess_sub, self.neighbors_det_sub]
		self.cb_list = [self.common_residual_callback, self.common_vel_callback, self.common_swt_callback, self.common_winbid_callback,\
						self.common_taskcomplete_callback, self.common_resetready_callback, self.common_resetcomplete_callback,\
						self.common_counting_callback, self.common_ess_callback, self.common_det_callback]

		self.uav0_hfov_pub = rospy.Publisher("/uav0/set_zoom", Float64, queue_size=100)

	def state_callback(self, msg):

		# Find all elements with the "uav" prefix and number less than 5
		agent_elements =\
					[element for element in msg.name if element.startswith('uav') and int(re.search(r'\d+', element).group()) < 5]
		# print("agent_elements: ", agent_elements)

		# Sort the list based on the numeric part of the elements
		agent_elements_sorted = sorted(agent_elements, key=lambda x: int(re.search(r'\d+', x).group()))
		# print("agent_elements_sorted: ", agent_elements_sorted)
		self.agents_name = agent_elements_sorted

		agents_index = [msg.name.index(element) for element in agent_elements_sorted]

		# Find all elements with the "uav" prefix and number greater than 4
		target_elements =\
					[element for element in msg.name if element.startswith('uav') and int(re.search(r'\d+', element).group()) > 4]
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
			O = np.array([msg.pose[index].orientation.w, msg.pose[index].orientation.x,\
							msg.pose[index].orientation.y, msg.pose[index].orientation.z])
			
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

	def link_callback(self, msg):

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

	def reference_callback(self, msg):

		width = len(msg.data)
		col_ind = msg.data.index(-1e6)
		one_hop_neighbors = msg.data.index(-2e6)
		Pd = msg.data.index(-3e6)
		cluster_set = msg.data.index(-4e6)
		team = msg.data.index(-5e6)
		# print("col_ind: ", col_ind)
		# print("one_hop_neighbors: ", one_hop_neighbors)
		# print("Pd: ", Pd)
		# print("cluster_set: ", cluster_set)
		# print("team: ", team)

		rec = np.array(msg.data)
		self.col_ind = int(rec[col_ind+1])
		self.neighbor_index = list(map(int, rec[one_hop_neighbors+1:Pd]))
		self.Pd = rec[Pd+1:cluster_set]
		self.cluster_set = list(map(int, rec[cluster_set+1:team]))
		self.teammate = list(map(int, rec[team+1:width]))

		# print("self.col_ind: ", self.col_ind)
		# print("self.neighbor_index: ", self.neighbor_index)
		# print("self.Pd: ", self.Pd)
		# print("self.cluster_set: ", self.cluster_set)
		# print("self.teammate: ", self.teammate)

	def SetCameraZoom(self, zoom):

		output = Float64(data = zoom)

		self.uav0_hfov_pub.publish(output)
		self.FL_Curr = zoom*self.FL_1x

	def CameraInfo0_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo0 = np.array([fx, fy, cx, cy])

	def Recognition0_callback(self, msg):

		pixel0, img0 = [], []
		n = len(self.cluster_set)

		for element in msg.bounding_boxes:

			if element.Class == self.targetName:

				xmin = element.xmin
				xmax = element.xmax
				ymin = element.ymin
				ymax = element.ymax

				# Pixel frame
				u = (xmin+xmax)/2
				v = (ymin+ymax)/2
				pixel0.append(np.array([u ,v]))
				# print("UAV0 - u&v: ", u, v)

				# Pixel frame convert to image plane (in meter)
				img0_hold = np.array([(u-self.camerainfo0[2]), (v-self.camerainfo0[3]), (self.FL_Curr)])
				# print("img0: ", self.img0)
				img0_hold /= np.linalg.norm(img0_hold)
				# print("img0 unit: ", self.img0)
				img0.append(img0_hold)

		if len(pixel0) == n:

			self.pixel0 = np.array(pixel0)
			self.img0 = np.array(img0)
		elif len(pixel0) < n and not np.array_equal(self.pixel0, np.array([[0.0,0.0,0.0]])) \
								and not np.array_equal(self.img0, np.array([[0.0,0.0,0.0]])):

			check_list = list(range(n))

			for point_B in pixel0:

				min_distance = float('inf')
				closest_index = None

				for i, point_A in enumerate(self.pixel0):

					distance = np.linalg.norm(point_B-point_A)

					if distance < min_distance:

						min_distance = distance
						closest_index = i

				check_list.remove(closest_index)

			for index in check_list:

				pixel0.append(self.pixel0[index])
				img0.append(self.img0[index])

			self.pixel0 = np.array(pixel0)
			self.img0 = np.array(img0)

		# print("pixel0: ", self.pixel0)
		# print("img0: ", self.img0)
		# a = np.reshape(self.pixel0,(1,4))
		# b = np.reshape(self.img0,(1,6))
		# print("reshpae pixel0: ", a)
		# print("reshpae img0: ", b)
		# print("reshpae pixel0: ", np.reshape(a,(2,2)))
		# print("reshpae img0: ", np.reshape(b,(2,3)))

	def q2yaw(self, q):

		if isinstance(q, Quaternion):

			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q[0], q[1], q[2], q[3])
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad

	def add_subscriber(self, topic, type_, name, queue_size, buff_size, varb_dict, sub_dict, init_varb, callback):

		if topic not in sub_dict:
			
			sub = rospy.Subscriber(topic, type_, callback, queue_size=queue_size, buff_size=buff_size)

			sub_dict[topic] = sub
			varb_dict[name] = init_varb

	def remove_subscriber(self, topic, name, varb_dict, sub_dict):

		if topic in sub_dict:

			sub_dict[topic].unregister()
			del sub_dict[topic]
			del varb_dict[name]

	def Neighbor_Connection(self):

		neighbor_index = self.neighbor_index.copy()
		teammate = self.teammate.copy()

		# print("id, neighbor_index: ", self.id, neighbor_index)
		# print("id, teammate: ", self.id, teammate)

		neighbor_topics = ["/uav" + str(index) for index in neighbor_index]
		# print("id, neighbor_topics: ", self.id, neighbor_topics)

		current_neighbor_topics_set = set()
		neighbor_topics_to_add = set()
		neighbor_topics_to_remove = set()

		if (self.last_neighbor_topics is None):

			current_neighbor_topics_set = set(neighbor_topics)
			neighbor_topics_to_add = current_neighbor_topics_set
			neighbor_topics_to_remove = current_neighbor_topics_set - current_neighbor_topics_set
		else:

			current_neighbor_topics_set = set(neighbor_topics)
			neighbor_topics_to_add = current_neighbor_topics_set - self.last_neighbor_topics
			neighbor_topics_to_remove = self.last_neighbor_topics - current_neighbor_topics_set

		# Remove neighbor
		for neighbor_topic in neighbor_topics_to_remove:

			for i, ref_topic in enumerate(self.topic_list):

				name = str(''.join(filter(str.isdigit, neighbor_topic)))
				self.remove_subscriber(str(neighbor_topic)+str(ref_topic), name, self.varb_list[i], self.sub_list[i])

		# print("After remove neighbor")
		# print("id, varb_list: ", self.id, self.varb_list)
		# print("id, sub_list: ", self.id, self.sub_list)

		index = 0
		if len(teammate) > 1:

			index = 10
		else:

			index = 3

		topics_to_remove = set()
		topics_to_add = set()

		if (self.last_index is None):

			topics_to_add = self.topic_list.copy()[0:index]
			topics_to_remove = set(topics_to_add) - set(topics_to_add)
		elif (self.last_index != index):

			topics_to_add = self.topic_list.copy()[0:index]
			topics_to_remove = self.topic_list.copy()[index:10]

		# Remove topic
		for i, ref_topic in enumerate(topics_to_remove):

			j = i+index

			for neighbor_topic in neighbor_topics:

				name = str(''.join(filter(str.isdigit, neighbor_topic)))
				self.remove_subscriber(str(neighbor_topic)+str(ref_topic), name, self.varb_list[j], self.sub_list[j])

		# print("After remove topic")
		# print("id, varb_list: ", self.id, self.varb_list)
		# print("id, sub_list: ", self.id, self.sub_list)

		# Add neighbor
		topic_list = self.topic_list.copy()[0:index]
		for neighbor_topic in neighbor_topics_to_add:

			for i, ref_topic in enumerate(topic_list):

				if ref_topic == "/Counting":

					name = str(''.join(filter(str.isdigit, neighbor_topic)))
					self.add_subscriber(str(neighbor_topic)+str(ref_topic), Int64, name, 50, 52428800,\
										self.varb_list[int(i)], self.sub_list[int(i)], self.init_list[int(i)], self.cb_list[int(i)])
				else:

					name = str(''.join(filter(str.isdigit, neighbor_topic)))
					self.add_subscriber(str(neighbor_topic)+str(ref_topic), Float64MultiArray, name, 50, 52428800,\
										self.varb_list[int(i)], self.sub_list[int(i)], self.init_list[int(i)], self.cb_list[int(i)])

		# print("After add neighbor")
		# print("id, varb_list: ", self.id, self.varb_list)
		# print("id, sub_list: ", self.id, self.sub_list)

		# Add topic
		for i, ref_topic in enumerate(topics_to_add):

			for neighbor_topic in neighbor_topics:

				if ref_topic == "/Counting":

					name = str(''.join(filter(str.isdigit, neighbor_topic)))
					self.add_subscriber(str(neighbor_topic)+str(ref_topic), Int64, name, 50, 52428800,\
										self.varb_list[int(i)], self.sub_list[int(i)], self.init_list[int(i)], self.cb_list[int(i)])
				else:

					name = str(''.join(filter(str.isdigit, neighbor_topic)))
					self.add_subscriber(str(neighbor_topic)+str(ref_topic), Float64MultiArray, name, 50, 52428800,\
										self.varb_list[int(i)], self.sub_list[int(i)], self.init_list[int(i)], self.cb_list[int(i)])

		# print("After add topic")
		# print("id, varb_list: ", self.id, self.varb_list)
		# print("id, sub_list: ", self.id, self.sub_list)
		# print("id, init_list: ", self.id, init_list)
		# print("id, cb_list: ", self.id, cb_list)

		self.last_neighbor_topics = current_neighbor_topics_set
		self.last_index = index

		varb_list_copy = copy.deepcopy(self.varb_list)

		return neighbor_index, teammate, varb_list_copy

	def common_residual_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		n = len(msg.data)

		if n > 0:

			content = []

			for i in range(int(n / 2)):

				content.append([msg.data[2 * i], msg.data[2 * i + 1]])

				self.neighbors_residual[name] = np.array(content)
		else:

			self.neighbors_residual[name] = np.array([[None, None]])

	def common_vel_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_vel[name] = np.array([msg.data[0], msg.data[1]])

	def common_swt_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_swt[name] = np.array([msg.data[0], msg.data[1]])

	def common_winbid_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_winbid[name] = msg.data

	def common_taskcomplete_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_taskcomplete[name] = msg.data[0]

	def common_resetready_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_resetready[name] = msg.data[0]

	def common_resetcomplete_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_resetcomplete[name] = msg.data[0]

	def common_counting_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_counting[name] = msg.data

	def common_ess_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))
		self.neighbors_ess[name] = np.array(msg.data)

	def common_det_callback(self, msg):

		topic = msg._connection_header['topic']
		name = str(''.join(filter(str.isdigit, topic)))

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbors_det[name] = np.reshape(hold, (int(col/3),3))

	def Collect_Data(self, neighbor_index, teammate, varb_list_copy):

		# Targets States
		targets_2dpos = [ [(state["position"][0], state["position"][1]), 1, 10] for state in self.targets_state]
		# print("targets_2dpos: ", targets_2dpos)

		targets_3dpos = [ state["position"] for state in self.targets_state]
		# print("targets_3dpos: ", targets_3dpos)

		targets_speed = np.array([ state["velocity"][0:2] for state in self.targets_state])
		# print("targets_speed: ", targets_speed)

		self.targets = {"2DPosition": targets_2dpos, "3DPosition": targets_3dpos, "Vel": targets_speed}

		# Self States
		self.pos = np.array([self.agents_state[self.id]["position"][0], self.agents_state[self.id]["position"][1]])
		# print("self.pos: ", self.pos)

		self.current_heading = self.q2yaw(self.agents_state[self.id]["orientation"])
		theta = self.current_heading
		self.perspective = np.array([1*cos(theta), 1*sin(theta)])
		# print("self.current_heading: ", self.current_heading)
		# print("self.perspective: ", self.perspective)

		self.states = {"Position": self.pos, "Perspective": self.perspective, "NeighorIndex": neighbor_index,\
						"Pd": self.Pd, "ClusterSet": self.cluster_set, "ColInd": self.col_ind, "Teammate": teammate,\
						"CamPos": self.Cams_state[self.id]["position"], "CamOrientation": self.Cams_state[self.id]["orientation"],\
						"img": self.img0, "pixel": self.pixel0, "camerainfo": self.camerainfo0}

		# Neighbors States
		neighbors_position = {str(i): [self.agents_state[i]["position"][0], self.agents_state[i]["position"][1]]\
								for i in range(self.n) if i != self.id}
		# print("neighbors_position: ", neighbors_position)

		neighbors_Cam_position = {str(i): self.Cams_state[i]["position"] for i in range(self.n) if i != self.id}
		# print("neighbors_Cam_position: ", neighbors_Cam_position)

		neighbors_Cam_Orientation = {str(i): self.Cams_state[i]["orientation"] for i in range(self.n) if i != self.id}
		# print("neighbors_Cam_Orientation: ", neighbors_Cam_Orientation)

		neighbors_residual = varb_list_copy[0]
		# print("neighbors_residual: ", neighbors_residual)

		neighbors_vel = varb_list_copy[1]
		# print("neighbors_vel: ", neighbors_vel)

		neighbors_id = neighbor_index
		# print("neighbors_id: ", neighbors_id)

		neighbors_swt = varb_list_copy[2]
		# print("neighbors_swt: ", neighbors_swt)

		neighbors_winbid = varb_list_copy[3]
		# print("neighbors_winbid: ", neighbors_winbid)

		neighbors_taskcomplete = varb_list_copy[4]
		# print("neighbors_taskcomplete: ", neighbors_taskcomplete)

		neighbors_resetready = varb_list_copy[5]
		# print("neighbors_resetready: ", neighbors_resetready)

		neighbors_resetcomplete = varb_list_copy[6]
		# print("neighbors_resetcomplete: ", neighbors_resetcomplete)

		neighbors_counting = varb_list_copy[7]
		# print("neighbors_counting: ", neighbors_counting)

		neighbors_Essential = varb_list_copy[8]
		# print("neighbors_Essential: ", neighbors_Essential)

		neighbors_Detection = varb_list_copy[9]
		# print("neighbors_Detection: ", neighbors_Detection)

		self.neighbors = {"Position": neighbors_position, "Residual": neighbors_residual, "Vel": neighbors_vel,\
							"ID": neighbors_id, "Swt": neighbors_swt,\
							"WinBid": neighbors_winbid, "TaskComplete": neighbors_taskcomplete, "ResetReady": neighbors_resetready,\
							"ResetComplete": neighbors_resetcomplete, "Counting": neighbors_counting,\
							"CamPos": neighbors_Cam_position, "CamOrientation": neighbors_Cam_Orientation,\
							"Essential": neighbors_Essential, "Detection": neighbors_Detection}

		# print("self.id, Target Information: ", self.id, self.targets)
		# print("self.id, Self Information: ", self.id, self.states)
		# print("self.id, Neighbor Information: ", self.id, self.neighbors)

	def	controller(self, dx, dp, step):

		self.cmd_vel.linear.x = 1.5*dx[0]*step
		self.cmd_vel.linear.y = 1.5*dx[1]*step
		self.cmd_vel.linear.z = 11.0 - self.agents_state[self.id]["position"][2]

		persp_t = self.perspective + dp*step
		persp_t /= np.linalg.norm(persp_t)

		axis = np.cross(self.perspective, persp_t)
		axis = axis/np.linalg.norm(axis)
		dot_product = np.dot(self.perspective, persp_t)
		dtheta = np.arccos( dot_product/(np.linalg.norm(self.perspective) * np.linalg.norm(persp_t)) )

		# print("dtheta: ", dtheta)

		self.cmd_vel.angular.z = 20.0*dtheta*axis

		self.px4.vel_control(self.cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_0')
		rate = rospy.Rate(100)

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		camera0 = { 'id'            :  0,
					'ecode'         :  "/uav0",
					'position'      :  np.array([2.0, 2.0]),
					'perspective'   :  np.array([1.0, 1.0]),
					'AngleofView'   :  20,
					'range_limit'   :  4.0,
					'lambda'        :  2,
					'color'         : (200, 0, 0)}

		ptz_0 = PTZcon(camera0, map_size, grid_size)
		uav_0 = UAV(camera0["id"])
		# est_0 = DepthEstimator(camera0['ecode'], "/home/leo/mts/src/coverage/record/uav0.csv",\
		# 					uav_0.targetName, uav_0.pixel_size, uav_0.image_width, uav_0.FL_Curr)

		# filename = "/home/leo/mts/src/coverage/record/uav0.csv"
		# f = open(filename, "w+")
		# f.close()

		while (uav_0.agents_state is None) or (uav_0.agents_name is None) or\
				(uav_0.targets_state is None) or (uav_0.targets_name is None) or\
				(uav_0.Cams_state is None) or (uav_0.Cams_name is None) or\
				(uav_0.col_ind is None) or (uav_0.neighbor_index is None) or (uav_0.Pd is None) or\
				(uav_0.cluster_set is None) or (uav_0.teammate is None):

			rate.sleep()

		last = time()

		while not rospy.is_shutdown():

			past = time()

			neighbor_index, teammate, varb_list_copy = uav_0.Neighbor_Connection()
			uav_0.Collect_Data(neighbor_index, teammate, varb_list_copy)

			dx, dp, step = ptz_0.UpdateState(uav_0.targets, uav_0.neighbors, uav_0.states,\
												np.round(time() - last, 2))
			uav_0.controller(dx, dp, step)
			
			# est_0.MultiTarget_Estimation(uav_0.states, uav_0.neighbors)
			# uav_0.SetCameraZoom(1)

			print("Calculation Time 0: " + str(time() - past))

			rate.sleep()

	except rospy.ROSInterruptException:
		pass