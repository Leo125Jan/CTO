#!/usr/bin/python3

import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
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
		self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o = None, None, None, None, None, None
		self.P3, self.P3o, self.P4, self.P4o  = None, None, None, None

		self.P5, self.P5v, self.P6, self.P6v, self.P7, self.P7v = None, None, None, None, None, None
		self.P8, self.P8v, self.P9, self.P9v = None, None, None, None

		self.Cam0, self.Cam1, self.Cam2, self.Cam3, self.Cam4 = None, None, None, None, None
		self.Cam0o, self.Cam1o, self.Cam2o, self.Cam3o, self.Cam4o = None, None, None, None, None
		
		self.cmd_vel = Twist()
		self.px4 = Px4Controller("uav4")

		size = int(25/0.1)
		self.current_heading = 0.0
		self.targetName = "Drone"
		self.pixel_size = 2.9*1e-6
		self.image_width = 1280
		self.FL_1x = 1280.785 # 6.5mm/2.9um
		self.FL_Curr = 1280.785
		self.img4, self.pixel4, self.camerainfo4 = np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0]])

		self.neighbor0_residual = np.array([[None, None]])
		self.neighbor1_residual = np.array([[None, None]])
		self.neighbor2_residual = np.array([[None, None]])
		self.neighbor3_residual = np.array([[None, None]])

		self.neighbor0_vel = np.array([0.0, 0.0])
		self.neighbor1_vel = np.array([0.0, 0.0])
		self.neighbor2_vel = np.array([0.0, 0.0])
		self.neighbor3_vel = np.array([0.0, 0.0])

		self.neighbor0_swt = np.array([0.0, 0.0])
		self.neighbor1_swt = np.array([0.0, 0.0])
		self.neighbor2_swt = np.array([0.0, 0.0])
		self.neighbor3_swt = np.array([0.0, 0.0])

		self.neighbor0_winbid = None
		self.neighbor1_winbid = None
		self.neighbor2_winbid = None
		self.neighbor3_winbid = None

		self.neighbor0_taskcomplete = False
		self.neighbor1_taskcomplete = False
		self.neighbor2_taskcomplete = False
		self.neighbor3_taskcomplete = False

		self.neighbor0_resetready = False
		self.neighbor1_resetready = False
		self.neighbor2_resetready = False
		self.neighbor3_resetready = False

		self.neighbor0_resetcomplete = False
		self.neighbor1_resetcomplete = False
		self.neighbor2_resetcomplete = False
		self.neighbor3_resetcomplete = False

		self.neighbor0_counting = None
		self.neighbor1_counting = None
		self.neighbor2_counting = None
		self.neighbor3_counting = None

		self.neighbor0_ess = None
		self.neighbor1_ess = None
		self.neighbor2_ess = None
		self.neighbor3_ess = None

		self.neighbor0_det = None
		self.neighbor1_det = None
		self.neighbor2_det = None
		self.neighbor3_det = None

		self.neighbor_index = None
		self.teammate = None

		# Publisher & Subscriber
		self.I_sub = rospy.Subscriber('/uav/StateI', ModelStates, self.state_callback, queue_size = 20, buff_size = 52428800)
		self.Link_sub = rospy.Subscriber('/uav/LinkI', LinkStates, self.link_callback, queue_size = 20, buff_size = 52428800)
		self.Ref_sub = rospy.Subscriber('/uav4/Ref', Float64MultiArray, self.reference_callback, queue_size = 20, buff_size = 52428800)

		self.uav4_camerainfo_sub = rospy.Subscriber("/uav4/camera/camera/color/camera_info", CameraInfo, self.CameraInfo4_callback, queue_size = 100)
		self.uav4_recognition_sub = rospy.Subscriber("/uav4/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition4_callback, queue_size = 100)

		self.neighbor0_residual_sub = rospy.Subscriber('/uav0/targetResdiual', Float64MultiArray, self.residual0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_residual_sub = rospy.Subscriber('/uav1/targetResdiual', Float64MultiArray, self.residual1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_residual_sub = rospy.Subscriber('/uav2/targetResdiual', Float64MultiArray, self.residual2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_residual_sub = rospy.Subscriber('/uav3/targetResdiual', Float64MultiArray, self.residual3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_vel_sub = rospy.Subscriber('/uav0/CurrVel', Float64MultiArray, self.vel0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_vel_sub = rospy.Subscriber('/uav1/CurrVel', Float64MultiArray, self.vel1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_vel_sub = rospy.Subscriber('/uav2/CurrVel', Float64MultiArray, self.vel2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_vel_sub = rospy.Subscriber('/uav3/CurrVel', Float64MultiArray, self.vel3_callback, queue_size = 20, buff_size = 52428800)
		
		self.neighbor0_swt_sub = rospy.Subscriber('/uav0/SweetSpot', Float64MultiArray, self.swt0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_swt_sub = rospy.Subscriber('/uav1/SweetSpot', Float64MultiArray, self.swt1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_swt_sub = rospy.Subscriber('/uav2/SweetSpot', Float64MultiArray, self.swt2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_swt_sub = rospy.Subscriber('/uav3/SweetSpot', Float64MultiArray, self.swt3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_winbid_sub = rospy.Subscriber('/uav0/WinBid', Float64MultiArray, self.winbid0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_winbid_sub = rospy.Subscriber('/uav1/WinBid', Float64MultiArray, self.winbid1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_winbid_sub = rospy.Subscriber('/uav2/WinBid', Float64MultiArray, self.winbid2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_winbid_sub = rospy.Subscriber('/uav3/WinBid', Float64MultiArray, self.winbid3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_taskcomplete_sub = rospy.Subscriber('/uav0/TaskComplete', Float64MultiArray, self.taskcomplete0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_taskcomplete_sub = rospy.Subscriber('/uav1/TaskComplete', Float64MultiArray, self.taskcomplete1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_taskcomplete_sub = rospy.Subscriber('/uav2/TaskComplete', Float64MultiArray, self.taskcomplete2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_taskcomplete_sub = rospy.Subscriber('/uav3/TaskComplete', Float64MultiArray, self.taskcomplete3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_resetready_sub = rospy.Subscriber('/uav0/ResetReady', Float64MultiArray, self.resetready0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_resetready_sub = rospy.Subscriber('/uav1/ResetReady', Float64MultiArray, self.resetready1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_resetready_sub = rospy.Subscriber('/uav2/ResetReady', Float64MultiArray, self.resetready2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_resetready_sub = rospy.Subscriber('/uav3/ResetReady', Float64MultiArray, self.resetready3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_resetcomplete_sub = rospy.Subscriber('/uav0/ResetComplete', Float64MultiArray, self.resetcomplete0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_resetcomplete_sub = rospy.Subscriber('/uav1/ResetComplete', Float64MultiArray, self.resetcomplete1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_resetcomplete_sub = rospy.Subscriber('/uav2/ResetComplete', Float64MultiArray, self.resetcomplete2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_resetcomplete_sub = rospy.Subscriber('/uav3/ResetComplete', Float64MultiArray, self.resetcomplete3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_counting_sub = rospy.Subscriber('/uav0/Counting', Int64, self.counting0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_counting_sub = rospy.Subscriber('/uav1/Counting', Int64, self.counting1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_counting_sub = rospy.Subscriber('/uav2/Counting', Int64, self.counting2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_counting_sub = rospy.Subscriber('/uav3/Counting', Int64, self.counting3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_ess_sub = rospy.Subscriber('/uav0/Essential', Float64MultiArray, self.ess0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_ess_sub = rospy.Subscriber('/uav1/Essential', Float64MultiArray, self.ess1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_ess_sub = rospy.Subscriber('/uav2/Essential', Float64MultiArray, self.ess2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_ess_sub = rospy.Subscriber('/uav3/Essential', Float64MultiArray, self.ess3_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_det_sub = rospy.Subscriber('/uav0/Detection', Float64MultiArray, self.det0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_det_sub = rospy.Subscriber('/uav1/Detection', Float64MultiArray, self.det1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor2_det_sub = rospy.Subscriber('/uav2/Detection', Float64MultiArray, self.det2_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_det_sub = rospy.Subscriber('/uav3/Detection', Float64MultiArray, self.det3_callback, queue_size = 20, buff_size = 52428800)


		self.uav4_hfov_pub = rospy.Publisher("/uav4/set_zoom", Float64, queue_size=100)

	def state_callback(self, msg):
		
		UAVs_index = [msg.name.index('uav0'), msg.name.index('uav1'), msg.name.index('uav2'), msg.name.index('uav3'),\
					msg.name.index('uav4'), msg.name.index('uav5'), msg.name.index('uav6'), msg.name.index('uav7'),\
					msg.name.index('uav8'), msg.name.index('uav9')]

		P0 = np.array([msg.pose[UAVs_index[0]].position.x, msg.pose[UAVs_index[0]].position.y, msg.pose[UAVs_index[0]].position.z])
		P0o = Quaternion([msg.pose[UAVs_index[0]].orientation.w, msg.pose[UAVs_index[0]].orientation.x,\
						msg.pose[UAVs_index[0]].orientation.y, msg.pose[UAVs_index[0]].orientation.z])

		P1 = np.array([msg.pose[UAVs_index[1]].position.x, msg.pose[UAVs_index[1]].position.y, msg.pose[UAVs_index[1]].position.z])
		P1o = Quaternion([msg.pose[UAVs_index[1]].orientation.w, msg.pose[UAVs_index[1]].orientation.x,\
						msg.pose[UAVs_index[1]].orientation.y, msg.pose[UAVs_index[1]].orientation.z])

		P2 = np.array([msg.pose[UAVs_index[2]].position.x, msg.pose[UAVs_index[2]].position.y, msg.pose[UAVs_index[2]].position.z])
		P2o = Quaternion([msg.pose[UAVs_index[2]].orientation.w, msg.pose[UAVs_index[2]].orientation.x,\
						msg.pose[UAVs_index[2]].orientation.y, msg.pose[UAVs_index[2]].orientation.z])

		P3 = np.array([msg.pose[UAVs_index[3]].position.x, msg.pose[UAVs_index[3]].position.y, msg.pose[UAVs_index[3]].position.z])
		P3o = Quaternion([msg.pose[UAVs_index[3]].orientation.w, msg.pose[UAVs_index[3]].orientation.x,\
						msg.pose[UAVs_index[3]].orientation.y, msg.pose[UAVs_index[3]].orientation.z])

		P4 = np.array([msg.pose[UAVs_index[4]].position.x, msg.pose[UAVs_index[4]].position.y, msg.pose[UAVs_index[4]].position.z])
		P4o = Quaternion([msg.pose[UAVs_index[4]].orientation.w, msg.pose[UAVs_index[4]].orientation.x,\
						msg.pose[UAVs_index[4]].orientation.y, msg.pose[UAVs_index[4]].orientation.z])

		P5 = np.array([msg.pose[UAVs_index[5]].position.x, msg.pose[UAVs_index[5]].position.y, msg.pose[UAVs_index[5]].position.z])
		P5v = np.array([msg.twist[UAVs_index[5]].linear.x, msg.twist[UAVs_index[5]].linear.y, msg.twist[UAVs_index[5]].linear.z])
		P6 = np.array([msg.pose[UAVs_index[6]].position.x, msg.pose[UAVs_index[6]].position.y, msg.pose[UAVs_index[6]].position.z])
		P6v = np.array([msg.twist[UAVs_index[6]].linear.x, msg.twist[UAVs_index[6]].linear.y, msg.twist[UAVs_index[6]].linear.z])
		P7 = np.array([msg.pose[UAVs_index[7]].position.x, msg.pose[UAVs_index[7]].position.y, msg.pose[UAVs_index[7]].position.z])
		P7v = np.array([msg.twist[UAVs_index[7]].linear.x, msg.twist[UAVs_index[7]].linear.y, msg.twist[UAVs_index[7]].linear.z])
		P8 = np.array([msg.pose[UAVs_index[8]].position.x, msg.pose[UAVs_index[8]].position.y, msg.pose[UAVs_index[8]].position.z])
		P8v = np.array([msg.twist[UAVs_index[8]].linear.x, msg.twist[UAVs_index[8]].linear.y, msg.twist[UAVs_index[8]].linear.z])
		P9 = np.array([msg.pose[UAVs_index[9]].position.x, msg.pose[UAVs_index[9]].position.y, msg.pose[UAVs_index[9]].position.z])
		P9v = np.array([msg.twist[UAVs_index[9]].linear.x, msg.twist[UAVs_index[9]].linear.y, msg.twist[UAVs_index[9]].linear.z])

		self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o = P0, P0o, P1, P1o, P2, P2o
		self.P3, self.P3o, self.P4, self.P4o = P3, P3o, P4, P4o
		
		self.P5, self.P5v, self.P6, self.P6v, self.P7, self.P7v = P5, P5v, P6, P6v, P7, P7v
		self.P8, self.P8v, self.P9, self.P9v = P8, P8v, P9, P9v

	def link_callback(self, msg):

		Link_index = [msg.name.index('uav0::cgo3_camera_link'), msg.name.index('uav1::cgo3_camera_link'),\
						msg.name.index('uav2::cgo3_camera_link'), msg.name.index('uav3::cgo3_camera_link'),\
						msg.name.index('uav4::cgo3_camera_link')]

		Cam0 = np.array([msg.pose[Link_index[0]].position.x, msg.pose[Link_index[0]].position.y, msg.pose[Link_index[0]].position.z])
		Cam0o = np.array([msg.pose[Link_index[0]].orientation.x, msg.pose[Link_index[0]].orientation.y,\
						msg.pose[Link_index[0]].orientation.z, msg.pose[Link_index[0]].orientation.w])

		Cam1 = np.array([msg.pose[Link_index[1]].position.x, msg.pose[Link_index[1]].position.y, msg.pose[Link_index[1]].position.z])
		Cam1o = np.array([msg.pose[Link_index[1]].orientation.x, msg.pose[Link_index[1]].orientation.y,\
						msg.pose[Link_index[1]].orientation.z, msg.pose[Link_index[1]].orientation.w])

		Cam2 = np.array([msg.pose[Link_index[2]].position.x, msg.pose[Link_index[2]].position.y, msg.pose[Link_index[2]].position.z])
		Cam2o = np.array([msg.pose[Link_index[2]].orientation.x, msg.pose[Link_index[2]].orientation.y,\
						msg.pose[Link_index[2]].orientation.z, msg.pose[Link_index[2]].orientation.w])

		Cam3 = np.array([msg.pose[Link_index[3]].position.x, msg.pose[Link_index[3]].position.y, msg.pose[Link_index[3]].position.z])
		Cam3o = np.array([msg.pose[Link_index[3]].orientation.x, msg.pose[Link_index[3]].orientation.y,\
						msg.pose[Link_index[3]].orientation.z, msg.pose[Link_index[3]].orientation.w])

		Cam4 = np.array([msg.pose[Link_index[4]].position.x, msg.pose[Link_index[4]].position.y, msg.pose[Link_index[4]].position.z])
		Cam4o = np.array([msg.pose[Link_index[4]].orientation.x, msg.pose[Link_index[4]].orientation.y,\
						msg.pose[Link_index[4]].orientation.z, msg.pose[Link_index[4]].orientation.w])

		self.Cam0, self.Cam1, self.Cam2, self.Cam3, self.Cam4 = Cam0, Cam1, Cam2, Cam3, Cam4
		self.Cam0o, self.Cam1o, self.Cam2o, self.Cam3o, self.Cam4o = Cam0o, Cam1o, Cam2o, Cam3o, Cam4o

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

		self.uav4_hfov_pub.publish(output)
		self.FL_Curr = zoom*self.FL_1x

	def CameraInfo4_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo4 = np.array([fx, fy, cx, cy])

	def Recognition4_callback(self, msg):

		pixel4, img4 = [], []
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
				pixel4.append(np.array([u ,v]))
				# print("UAV0 - u&v: ", u, v)

				# Pixel frame convert to image plane (in meter)
				img4_hold = np.array([(u-self.camerainfo4[2]), (v-self.camerainfo4[3]), (self.FL_Curr)])
				# print("img0: ", self.img0)
				img4_hold /= np.linalg.norm(img4_hold)
				# print("img0 unit: ", self.img0)
				img4.append(img4_hold)

		if len(pixel4) == n:

			self.pixel4 = np.array(pixel4)
			self.img4 = np.array(img4)
		elif len(pixel4) < n and not np.array_equal(self.pixel4, np.array([[0.0,0.0,0.0]])) \
								and not np.array_equal(self.img4, np.array([[0.0,0.0,0.0]])):

			check_list = list(range(n))

			for point_B in pixel4:

				min_distance = float('inf')
				closest_index = None

				for i, point_A in enumerate(self.pixel4):

					distance = np.linalg.norm(point_B-point_A)

					if distance < min_distance:

						min_distance = distance
						closest_index = i

				check_list.remove(closest_index)

			for index in check_list:

				pixel4.append(self.pixel4[index])
				img4.append(self.img4[index])

			self.pixel4 = np.array(pixel4)
			self.img4 = np.array(img4)

	def q2yaw(self, q):

		if isinstance(q, Quaternion):

			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q[0], q[1], q[2], q[3])
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad

	def residual0_callback(self, msg):

		n = len(msg.data)
		# print("Neighbor0_residual: ", msg.data)
		# print("n: ", n)

		if n > 0:

			content = []

			for i in range(int(n/2)):

				content.append([msg.data[2*i], msg.data[2*i+1]])

			self.neighbor0_residual = np.array(content)
		else:

			self.neighbor0_residual = np.array([[None, None]])

	def residual1_callback(self, msg):

		n = len(msg.data)
		# print("Neighbor1_residual: ", msg.data)
		# print("n: ", n)

		if n > 0:

			content = []

			for i in range(int(n/2)):

				content.append([msg.data[2*i], msg.data[2*i+1]])

			self.neighbor1_residual = np.array(content)
		else:

			self.neighbor1_residual = np.array([[None, None]])

	def residual2_callback(self, msg):

		n = len(msg.data)
		# print("Neighbor2_residual: ", msg.data)
		# print("n: ", n)

		if n > 0:

			content = []

			for i in range(int(n/2)):

				content.append([msg.data[2*i], msg.data[2*i+1]])

			self.neighbor2_residual = np.array(content)
		else:

			self.neighbor2_residual = np.array([[None, None]])

	def residual3_callback(self, msg):

		n = len(msg.data)
		# print("Neighbor3_residual: ", msg.data)
		# print("n: ", n)

		if n > 0:

			content = []

			for i in range(int(n/2)):

				content.append([msg.data[2*i], msg.data[2*i+1]])

			self.neighbor3_residual = np.array(content)
		else:

			self.neighbor3_residual = np.array([[None, None]])

	def vel0_callback(self, msg):

		self.neighbor0_vel = np.array([msg.data[0], msg.data[1]])

	def vel1_callback(self, msg):

		self.neighbor1_vel = np.array([msg.data[0], msg.data[1]])

	def vel2_callback(self, msg):

		self.neighbor2_vel = np.array([msg.data[0], msg.data[1]])

	def vel3_callback(self, msg):

		self.neighbor3_vel = np.array([msg.data[0], msg.data[1]])

	def swt0_callback(self, msg):

		self.neighbor0_swt = np.array([msg.data[0], msg.data[1]])

	def swt1_callback(self, msg):

		self.neighbor1_swt = np.array([msg.data[0], msg.data[1]])

	def swt2_callback(self, msg):

		self.neighbor2_swt = np.array([msg.data[0], msg.data[1]])

	def swt3_callback(self, msg):

		self.neighbor3_swt = np.array([msg.data[0], msg.data[1]])

	def winbid0_callback(self, msg):

		self.neighbor0_winbid = msg.data

	def winbid1_callback(self, msg):

		self.neighbor1_winbid = msg.data

	def winbid2_callback(self, msg):

		self.neighbor2_winbid = msg.data

	def winbid3_callback(self, msg):

		self.neighbor3_winbid = msg.data

	def taskcomplete0_callback(self, msg):

		self.neighbor0_taskcomplete = msg.data[0]

	def taskcomplete1_callback(self, msg):

		self.neighbor1_taskcomplete = msg.data[0]

	def taskcomplete2_callback(self, msg):

		self.neighbor2_taskcomplete = msg.data[0]

	def taskcomplete3_callback(self, msg):

		self.neighbor3_taskcomplete = msg.data[0]

	def resetready0_callback(self, msg):

		self.neighbor0_resetready = msg.data[0]

	def resetready1_callback(self, msg):

		self.neighbor1_resetready = msg.data[0]

	def resetready2_callback(self, msg):

		self.neighbor2_resetready = msg.data[0]

	def resetready3_callback(self, msg):

		self.neighbor3_resetready = msg.data[0]

	def resetcomplete0_callback(self, msg):

		self.neighbor0_resetcomplete = msg.data[0]

	def resetcomplete1_callback(self, msg):

		self.neighbor1_resetcomplete = msg.data[0]

	def resetcomplete2_callback(self, msg):

		self.neighbor2_resetcomplete = msg.data[0]

	def resetcomplete3_callback(self, msg):

		self.neighbor3_resetcomplete = msg.data[0]

	def counting0_callback(self, msg):

		self.neighbor0_counting = msg.data

	def counting1_callback(self, msg):

		self.neighbor1_counting = msg.data

	def counting2_callback(self, msg):

		self.neighbor2_counting = msg.data

	def counting3_callback(self, msg):

		self.neighbor3_counting = msg.data

	def ess0_callback(self, msg):

		self.neighbor0_ess = np.array(msg.data)

	def ess1_callback(self, msg):

		self.neighbor1_ess = np.array(msg.data)

	def ess2_callback(self, msg):

		self.neighbor2_ess = np.array(msg.data)

	def ess3_callback(self, msg):

		self.neighbor3_ess = np.array(msg.data)

	def det0_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor0_det = np.reshape(hold, (int(col/3),3))

	def det1_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor1_det = np.reshape(hold, (int(col/3),3))

	def det2_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor2_det = np.reshape(hold, (int(col/3),3))

	def det3_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor3_det = np.reshape(hold, (int(col/3),3))

	def Collect_Data(self):

		# Targets States 
		targets_2dpos = [[(self.P5[0], self.P5[1]), 1, 10], [(self.P6[0], self.P6[1]), 1, 10], [(self.P7[0], self.P7[1]), 1, 10],\
						[(self.P8[0], self.P8[1]), 1, 10], [(self.P9[0], self.P9[1]), 1, 10]]
		targets_3dpos = [self.P5, self.P6, self.P7, self.P8, self.P9]

		targets_speed = np.array([self.P5v[0:2], self.P6v[0:2], self.P7v[0:2], self.P8v[0:2], self.P9v[0:2]])
		self.targets = {"2DPosition": targets_2dpos, "3DPosition": targets_3dpos, "Vel": targets_speed}

		# Self States
		self.pos = np.array([self.P4[0], self.P4[1]])
		self.current_heading = self.q2yaw(self.P4o)
		theta = self.current_heading
		self.perspective = np.array([1*cos(theta), 1*sin(theta)])

		self.states = {"Position": self.pos, "Perspective": self.perspective, "NeighorIndex": self.neighbor_index,\
						"Pd": self.Pd, "ClusterSet": self.cluster_set, "ColInd": self.col_ind, "Teammate": self.teammate,\
						"CamPos": self.Cam4, "CamOrientation": self.Cam4o,\
						"img": self.img4, "pixel": self.pixel4, "camerainfo": self.camerainfo4}

		# Neighbors States
		neighbors_position = {"0": [self.P0[0], self.P0[1]], "1": [self.P1[0], self.P1[1]],\
								"2": [self.P2[0], self.P2[1]], "3": [self.P3[0], self.P3[1]]}
		neighbors_Cam_position = {"0": self.Cam0, "1": self.Cam1, "2": self.Cam2, "3": self.Cam3}
		neighbors_Cam_Orientation = {"0": self.Cam0o, "1": self.Cam1o, "2": self.Cam2o, "3": self.Cam3o}

		# print("Neighbor1_residual: ", self.neighbor1_residual)
		# print("Neighbor2_residual: ", self.neighbor2_residual)
		neighbors_residual = {"0": self.neighbor0_residual, "1": self.neighbor1_residual,\
								"2": self.neighbor2_residual, "3": self.neighbor3_residual}

		neighbors_vel = {"0": self.neighbor0_vel, "1": self.neighbor1_vel, "2": self.neighbor2_vel, "3": self.neighbor3_vel}
		neighbors_id = self.neighbor_index
		neighbors_swt = {"0": self.neighbor0_swt, "1": self.neighbor1_swt, "2": self.neighbor2_swt, "3": self.neighbor3_swt}
		neighbors_winbid = {"0": self.neighbor0_winbid, "1": self.neighbor1_winbid, "2": self.neighbor2_winbid, "3": self.neighbor3_winbid}
		neighbors_taskcomplete = {"0": self.neighbor0_taskcomplete, "1": self.neighbor1_taskcomplete, "2": self.neighbor2_taskcomplete,\
									"3": self.neighbor3_taskcomplete}
		neighbors_resetready = {"0": self.neighbor0_resetready, "1": self.neighbor1_resetready, "2": self.neighbor2_resetready,\
									"3": self.neighbor3_resetready}
		neighbors_resetcomplete = {"0": self.neighbor0_resetcomplete, "1": self.neighbor1_resetcomplete, "2": self.neighbor2_resetcomplete,\
									"3": self.neighbor3_resetcomplete}
		neighbors_counting = {"0": self.neighbor0_counting, "1": self.neighbor1_counting, "2": self.neighbor2_counting,\
									"3": self.neighbor3_counting}
		neighbors_Essential = {"0": self.neighbor0_ess, "1": self.neighbor1_ess, "2": self.neighbor2_ess, "3": self.neighbor3_ess}
		neighbors_Detection = {"0": self.neighbor0_det, "1": self.neighbor1_det, "2": self.neighbor2_det, "3": self.neighbor3_det}

		self.neighbors = {"Position": neighbors_position, "Residual": neighbors_residual, "Vel": neighbors_vel,\
							"ID": neighbors_id, "Swt": neighbors_swt,\
							"WinBid": neighbors_winbid, "TaskComplete": neighbors_taskcomplete, "ResetReady": neighbors_resetready,\
							"ResetComplete": neighbors_resetcomplete, "Counting": neighbors_counting,\
							"CamPos": neighbors_Cam_position, "CamOrientation": neighbors_Cam_Orientation,\
							"Essential": neighbors_Essential, "Detection": neighbors_Detection}

		# print("Target Information: ", self.targets)
		# print("Self Information: ", self.states)
		# print("Neighbor Information: ", self.neighbors)

	def	controller(self, dx, dp, step):

		self.cmd_vel.linear.x = 1.5*dx[0]*step
		self.cmd_vel.linear.y = 1.5*dx[1]*step
		self.cmd_vel.linear.z = 11.0 - self.P4[2]

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
		rospy.init_node('controller_4')
		rate = rospy.Rate(100)

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		camera4 = { 'id'            :  4,
					'ecode'         :  "/uav4",
					'position'      :  np.array([12.5, 2.0]),
					'perspective'   :  np.array([0.0, 1.0]),
					'AngleofView'   :  20,
					'range_limit'   :  4.0,
					'lambda'        :  2,
					'color'         : (200, 0, 0)}

		ptz_4 = PTZcon(camera4, map_size, grid_size)
		uav_4 = UAV(camera4["id"])
		# est_4 = DepthEstimator(camera4['ecode'], "/home/leo/mts/src/coverage/record/uav4.csv",\
		# 					uav_4.targetName, uav_4.pixel_size, uav_4.image_width, uav_4.FL_Curr)

		# filename = "/home/leo/mts/src/coverage/record/uav4.csv"
		# f = open(filename, "w+")
		# f.close()

		while (uav_4.P4 is None) or (uav_4.neighbor_index is None):

			rate.sleep()

		last = time()

		while not rospy.is_shutdown():

			past = time()
			uav_4.Collect_Data()
			dx, dp, step = ptz_4.UpdateState(uav_4.targets, uav_4.neighbors, uav_4.states,\
												np.round(time() - last, 2))
			uav_4.controller(dx, dp, step)
			# est_4.MultiTarget_Estimation(uav_4.states, uav_4.neighbors)
			# uav_4.SetCameraZoom(1)

			print("Calculation Time 4: " + str(time() - past))

			rate.sleep()

	except rospy.ROSInterruptException:
		pass
