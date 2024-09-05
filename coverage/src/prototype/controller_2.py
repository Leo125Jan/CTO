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

		# Variable of Control
		self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o = None, None, None, None, None, None
		self.P3, self.P3o, self.P4, self.P4o  = None, None, None, None

		self.P5, self.P5v, self.P6, self.P6v, self.P7, self.P7v = None, None, None, None, None, None
		self.P8, self.P8v, self.P9, self.P9v = None, None, None, None

		self.Cam0, self.Cam1, self.Cam2, self.Cam3, self.Cam4 = None, None, None, None, None
		self.Cam0o, self.Cam1o, self.Cam2o, self.Cam3o, self.Cam4o = None, None, None, None, None

		self.A, self.b = None, None
		self.cmd_vel = Twist()
		self.d_safe = 1.0
		self.m, self.x = None, None
		self.px4 = Px4Controller("uav2")

		size = int(25/0.1)
		self.current_heading = 0.0
		self.targetName = "Drone"
		self.pixel_size = 2.9*1e-6
		self.image_width = 1280
		self.FL_1x = 1280.785 # 6.5mm/2.9um
		self.FL_Curr = self.FL_1x
		self.img2, self.pixel2, self.camerainfo2 = np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0]]), np.array([[0.0,0.0,0.0,0.0]])

		self.neighbor0_residual = np.array([[None, None]])
		self.neighbor1_residual = np.array([[None, None]])
		self.neighbor3_residual = np.array([[None, None]])
		self.neighbor4_residual = np.array([[None, None]])

		self.neighbor0_vel = np.array([0.0, 0.0])
		self.neighbor1_vel = np.array([0.0, 0.0])
		self.neighbor3_vel = np.array([0.0, 0.0])
		self.neighbor4_vel = np.array([0.0, 0.0])

		self.neighbor0_swt = np.array([0.0, 0.0])
		self.neighbor1_swt = np.array([0.0, 0.0])
		self.neighbor3_swt = np.array([0.0, 0.0])
		self.neighbor4_swt = np.array([0.0, 0.0])

		self.neighbor0_winbid = None
		self.neighbor1_winbid = None
		self.neighbor3_winbid = None
		self.neighbor4_winbid = None

		self.neighbor0_taskcomplete = False
		self.neighbor1_taskcomplete = False
		self.neighbor3_taskcomplete = False
		self.neighbor4_taskcomplete = False

		self.neighbor0_resetready = False
		self.neighbor1_resetready = False
		self.neighbor3_resetready = False
		self.neighbor4_resetready = False

		self.neighbor0_resetcomplete = False
		self.neighbor1_resetcomplete = False
		self.neighbor3_resetcomplete = False
		self.neighbor4_resetcomplete = False

		self.neighbor0_counting = None
		self.neighbor1_counting = None
		self.neighbor3_counting = None
		self.neighbor4_counting = None

		self.neighbor0_ess = None
		self.neighbor1_ess = None
		self.neighbor3_ess = None
		self.neighbor4_ess = None

		self.neighbor0_det = None
		self.neighbor1_det = None
		self.neighbor3_det = None
		self.neighbor4_det = None

		self.neighbor_index = None
		self.teammate = None

		# Publisher & Subscriber
		self.States_sub = rospy.Subscriber('/uav/StateI', ModelStates, self.state_callback, queue_size = 20, buff_size = 52428800)
		self.Link_sub = rospy.Subscriber('/uav/LinkI', LinkStates, self.link_callback, queue_size = 20, buff_size = 52428800)
		self.Ref_sub = rospy.Subscriber('/uav2/Ref', Float64MultiArray, self.reference_callback, queue_size = 20, buff_size = 52428800)

		self.uav2_camerainfo_sub = rospy.Subscriber("/uav2/camera/camera/color/camera_info", CameraInfo, self.CameraInfo2_callback, queue_size = 100)
		self.uav2_recognition_sub = rospy.Subscriber("/uav2/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition2_callback, queue_size = 100)

		self.neighbor0_residual_sub = rospy.Subscriber('/uav0/targetResdiual', Float64MultiArray, self.residual0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_residual_sub = rospy.Subscriber('/uav1/targetResdiual', Float64MultiArray, self.residual1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_residual_sub = rospy.Subscriber('/uav3/targetResdiual', Float64MultiArray, self.residual3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_residual_sub = rospy.Subscriber('/uav4/targetResdiual', Float64MultiArray, self.residual4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_vel_sub = rospy.Subscriber('/uav0/CurrVel', Float64MultiArray, self.vel0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_vel_sub = rospy.Subscriber('/uav1/CurrVel', Float64MultiArray, self.vel1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_vel_sub = rospy.Subscriber('/uav3/CurrVel', Float64MultiArray, self.vel3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_vel_sub = rospy.Subscriber('/uav4/CurrVel', Float64MultiArray, self.vel4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_swt_sub = rospy.Subscriber('/uav0/SweetSpot', Float64MultiArray, self.swt0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_swt_sub = rospy.Subscriber('/uav1/SweetSpot', Float64MultiArray, self.swt1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_swt_sub = rospy.Subscriber('/uav3/SweetSpot', Float64MultiArray, self.swt3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_swt_sub = rospy.Subscriber('/uav4/SweetSpot', Float64MultiArray, self.swt4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_winbid_sub = rospy.Subscriber('/uav0/WinBid', Float64MultiArray, self.winbid0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_winbid_sub = rospy.Subscriber('/uav1/WinBid', Float64MultiArray, self.winbid1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_winbid_sub = rospy.Subscriber('/uav3/WinBid', Float64MultiArray, self.winbid3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_winbid_sub = rospy.Subscriber('/uav4/WinBid', Float64MultiArray, self.winbid4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_taskcomplete_sub = rospy.Subscriber('/uav0/TaskComplete', Float64MultiArray, self.taskcomplete0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_taskcomplete_sub = rospy.Subscriber('/uav1/TaskComplete', Float64MultiArray, self.taskcomplete1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_taskcomplete_sub = rospy.Subscriber('/uav3/TaskComplete', Float64MultiArray, self.taskcomplete3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_taskcomplete_sub = rospy.Subscriber('/uav4/TaskComplete', Float64MultiArray, self.taskcomplete4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_resetready_sub = rospy.Subscriber('/uav0/ResetReady', Float64MultiArray, self.resetready0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_resetready_sub = rospy.Subscriber('/uav1/ResetReady', Float64MultiArray, self.resetready1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_resetready_sub = rospy.Subscriber('/uav3/ResetReady', Float64MultiArray, self.resetready3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_resetready_sub = rospy.Subscriber('/uav4/ResetReady', Float64MultiArray, self.resetready4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_resetcomplete_sub = rospy.Subscriber('/uav0/ResetComplete', Float64MultiArray, self.resetcomplete0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_resetcomplete_sub = rospy.Subscriber('/uav1/ResetComplete', Float64MultiArray, self.resetcomplete1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_resetcomplete_sub = rospy.Subscriber('/uav3/ResetComplete', Float64MultiArray, self.resetcomplete3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_resetcomplete_sub = rospy.Subscriber('/uav4/ResetComplete', Float64MultiArray, self.resetcomplete4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_counting_sub = rospy.Subscriber('/uav0/Counting', Int64, self.counting0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_counting_sub = rospy.Subscriber('/uav1/Counting', Int64, self.counting1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_counting_sub = rospy.Subscriber('/uav3/Counting', Int64, self.counting3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_counting_sub = rospy.Subscriber('/uav4/Counting', Int64, self.counting4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_ess_sub = rospy.Subscriber('/uav0/Essential', Float64MultiArray, self.ess0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_ess_sub = rospy.Subscriber('/uav1/Essential', Float64MultiArray, self.ess1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_ess_sub = rospy.Subscriber('/uav3/Essential', Float64MultiArray, self.ess3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_ess_sub = rospy.Subscriber('/uav4/Essential', Float64MultiArray, self.ess4_callback, queue_size = 20, buff_size = 52428800)

		self.neighbor0_det_sub = rospy.Subscriber('/uav0/Detection', Float64MultiArray, self.det0_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor1_det_sub = rospy.Subscriber('/uav1/Detection', Float64MultiArray, self.det1_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor3_det_sub = rospy.Subscriber('/uav3/Detection', Float64MultiArray, self.det3_callback, queue_size = 20, buff_size = 52428800)
		self.neighbor4_det_sub = rospy.Subscriber('/uav4/Detection', Float64MultiArray, self.det4_callback, queue_size = 20, buff_size = 52428800)

		self.uav2_hfov_pub = rospy.Publisher("/uav2/set_zoom", Float64, queue_size=100)

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

		PO = np.array([100, 100, 100]); self.PO = PO
		self.A = np.array([(-2*(P3-PO)[:2]).tolist()])
		self.b = np.array([np.linalg.norm((P3-PO)[:2])**2 - self.d_safe**2])

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

	def q2yaw(self, q):

		if isinstance(q, Quaternion):

			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q[0], q[1], q[2], q[3])
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return rotate_z_rad

	def SetCameraZoom(self, zoom):

		output = Float64(data = zoom)

		self.uav2_hfov_pub.publish(output)
		self.FL_Curr = zoom*self.FL_1x

	def Recognition2_callback(self, msg):

		pixel2, img2 = [], []
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
				pixel2.append(np.array([u ,v]))
				# print("UAV0 - u&v: ", u, v)

				# Pixel frame convert to image plane (in meter)
				img2_hold = np.array([(u-self.camerainfo2[2]), (v-self.camerainfo2[3]), (self.FL_Curr)])
				# print("img0: ", self.img0)
				img2_hold /= np.linalg.norm(img2_hold)
				# print("img0 unit: ", self.img0)
				img2.append(img2_hold)

		if len(pixel2) == n:

			self.pixel2 = np.array(pixel2)
			self.img2 = np.array(img2)
		elif len(pixel2) < n and not np.array_equal(self.pixel2, np.array([[0.0,0.0,0.0]])) \
								and not np.array_equal(self.img2, np.array([[0.0,0.0,0.0]])):

			check_list = list(range(n))

			for point_B in pixel2:

				min_distance = float('inf')
				closest_index = None

				for i, point_A in enumerate(self.pixel2):

					distance = np.linalg.norm(point_B-point_A)

					if distance < min_distance:

						min_distance = distance
						closest_index = i

				check_list.remove(closest_index)

			for index in check_list:

				pixel2.append(self.pixel2[index])
				img2.append(self.img2[index])

			self.pixel2 = np.array(pixel2)
			self.img2 = np.array(img2)

	def CameraInfo2_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo2 = np.array([fx, fy, cx, cy])

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

	def residual4_callback(self, msg):

		n = len(msg.data)
		# print("Neighbor4_residual: ", msg.data)
		# print("n: ", n)

		if n > 0:

			content = []

			for i in range(int(n/2)):

				content.append([msg.data[2*i], msg.data[2*i+1]])

			self.neighbor4_residual = np.array(content)
		else:

			self.neighbor4_residual = np.array([[None, None]])

	def vel0_callback(self, msg):

		self.neighbor0_vel = np.array([msg.data[0], msg.data[1]])

	def vel1_callback(self, msg):

		self.neighbor1_vel = np.array([msg.data[0], msg.data[1]])

	def vel3_callback(self, msg):

		self.neighbor3_vel = np.array([msg.data[0], msg.data[1]])

	def vel4_callback(self, msg):

		self.neighbor4_vel = np.array([msg.data[0], msg.data[1]])

	def swt0_callback(self, msg):

		self.neighbor0_swt = np.array([msg.data[0], msg.data[1]])

	def swt1_callback(self, msg):

		self.neighbor1_swt = np.array([msg.data[0], msg.data[1]])

	def swt3_callback(self, msg):

		self.neighbor3_swt = np.array([msg.data[0], msg.data[1]])

	def swt4_callback(self, msg):

		self.neighbor4_swt = np.array([msg.data[0], msg.data[1]])

	def winbid0_callback(self, msg):

		self.neighbor0_winbid = msg.data

	def winbid1_callback(self, msg):

		self.neighbor1_winbid = msg.data

	def winbid3_callback(self, msg):

		self.neighbor3_winbid = msg.data

	def winbid4_callback(self, msg):

		self.neighbor4_winbid = msg.data

	def taskcomplete0_callback(self, msg):

		self.neighbor0_taskcomplete = msg.data[0]

	def taskcomplete1_callback(self, msg):

		self.neighbor1_taskcomplete = msg.data[0]

	def taskcomplete3_callback(self, msg):

		self.neighbor3_taskcomplete = msg.data[0]

	def taskcomplete4_callback(self, msg):

		self.neighbor4_taskcomplete = msg.data[0]

	def resetready0_callback(self, msg):

		self.neighbor0_resetready = msg.data[0]

	def resetready1_callback(self, msg):

		self.neighbor1_resetready = msg.data[0]

	def resetready3_callback(self, msg):

		self.neighbor3_resetready = msg.data[0]

	def resetready4_callback(self, msg):

		self.neighbor4_resetready = msg.data[0]

	def resetcomplete0_callback(self, msg):

		self.neighbor0_resetcomplete = msg.data[0]

	def resetcomplete1_callback(self, msg):

		self.neighbor1_resetcomplete = msg.data[0]

	def resetcomplete3_callback(self, msg):

		self.neighbor3_resetcomplete = msg.data[0]

	def resetcomplete4_callback(self, msg):

		self.neighbor4_resetcomplete = msg.data[0]

	def counting0_callback(self, msg):

		self.neighbor0_counting = msg.data

	def counting1_callback(self, msg):

		self.neighbor1_counting = msg.data

	def counting3_callback(self, msg):

		self.neighbor3_counting = msg.data

	def counting4_callback(self, msg):

		self.neighbor4_counting = msg.data

	def ess0_callback(self, msg):

		self.neighbor0_ess = msg.data

	def ess1_callback(self, msg):

		self.neighbor1_ess = msg.data

	def ess3_callback(self, msg):

		self.neighbor3_ess = msg.data

	def ess4_callback(self, msg):

		self.neighbor4_ess = msg.data

	def det0_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor0_det = np.reshape(hold, (int(col/3),3))

	def det1_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor1_det = np.reshape(hold, (int(col/3),3))

	def det3_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor3_det = np.reshape(hold, (int(col/3),3))

	def det4_callback(self, msg):

		hold = np.array(msg.data)
		col = np.shape(hold)[0]
		self.neighbor4_det = np.reshape(hold, (int(col/3),3))

	def Collect_Data(self):

		# Targets States 
		targets_2dpos = [[(self.P5[0], self.P5[1]), 1, 10], [(self.P6[0], self.P6[1]), 1, 10], [(self.P7[0], self.P7[1]), 1, 10],\
						[(self.P8[0], self.P8[1]), 1, 10], [(self.P9[0], self.P9[1]), 1, 10]]
		targets_3dpos = [self.P5, self.P6, self.P7, self.P8, self.P9]

		targets_speed = np.array([self.P5v[0:2], self.P6v[0:2], self.P7v[0:2], self.P8v[0:2], self.P9v[0:2]])
		self.targets = {"2DPosition": targets_2dpos, "3DPosition": targets_3dpos, "Vel": targets_speed}

		# Self States
		self.pos = np.array([self.P2[0], self.P2[1]])
		self.current_heading = self.q2yaw(self.P2o)
		theta = self.current_heading
		self.perspective = np.array([1*cos(theta), 1*sin(theta)])

		self.states = {"Position": self.pos, "Perspective": self.perspective, "NeighorIndex": self.neighbor_index,\
						"Pd": self.Pd, "ClusterSet": self.cluster_set, "ColInd": self.col_ind, "Teammate": self.teammate,\
						"CamPos": self.Cam2, "CamOrientation": self.Cam2o,\
						"img": self.img2, "pixel": self.pixel2, "camerainfo": self.camerainfo2}

		# Neighbors States
		neighbors_position = {"0": [self.P0[0], self.P0[1]], "1": [self.P1[0], self.P1[1]],\
								"3": [self.P3[0], self.P3[1]], "4": [self.P4[0], self.P4[1]]}
		neighbors_Cam_position = {"0": self.Cam0, "1": self.Cam1, "3":self.Cam3, "4": self.Cam4}
		neighbors_Cam_Orientation = {"0": self.Cam0o, "1": self.Cam1o, "3":self.Cam3o, "4": self.Cam4o}

		# print("Neighbor0_residual: ", self.neighbor0_residual)
		# print("Neighbor1_residual: ", self.neighbor1_residual)
		neighbors_residual = {"0": self.neighbor0_residual, "1": self.neighbor1_residual,\
								"3": self.neighbor3_residual, "4": self.neighbor4_residual}
		neighbors_vel = {"0": self.neighbor0_vel, "1": self.neighbor1_vel, "3": self.neighbor3_vel, "4": self.neighbor4_vel}
		neighbors_id = self.neighbor_index
		neighbors_swt = {"0": self.neighbor0_swt, "1": self.neighbor1_swt, "3": self.neighbor3_swt, "4": self.neighbor4_swt}
		neighbors_winbid = {"0": self.neighbor0_winbid, "1": self.neighbor1_winbid, "3": self.neighbor3_winbid, "4": self.neighbor4_winbid}
		neighbors_taskcomplete = {"0": self.neighbor0_taskcomplete, "1": self.neighbor1_taskcomplete, "3": self.neighbor3_taskcomplete,\
									"4": self.neighbor4_taskcomplete}
		neighbors_resetready = {"0": self.neighbor0_resetready, "1": self.neighbor1_resetready, "3": self.neighbor3_resetready,\
								"4": self.neighbor4_resetready}
		neighbors_resetcomplete = {"0": self.neighbor0_resetcomplete, "1": self.neighbor1_resetcomplete, "3": self.neighbor3_resetcomplete,\
									"4": self.neighbor4_resetcomplete}
		neighbors_counting = {"0": self.neighbor0_counting, "1": self.neighbor1_counting, "3": self.neighbor3_counting,\
									"4": self.neighbor4_counting}
		neighbors_Essential = {"0": self.neighbor0_ess, "1": self.neighbor1_ess, "3": self.neighbor3_ess, "4": self.neighbor4_ess}
		neighbors_Detection = {"0": self.neighbor0_det, "1": self.neighbor1_det, "3": self.neighbor3_det, "4": self.neighbor4_det}

		self.neighbors = {"Position": neighbors_position, "Residual": neighbors_residual, "Vel": neighbors_vel,\
							"ID": neighbors_id, "Swt": neighbors_swt,\
							"WinBid": neighbors_winbid, "TaskComplete": neighbors_taskcomplete, "ResetReady": neighbors_resetready,\
							"ResetComplete": neighbors_resetcomplete, "Counting": neighbors_counting,\
							"CamPos": neighbors_Cam_position, "CamOrientation": neighbors_Cam_Orientation,\
							"Essential": neighbors_Essential, "Detection": neighbors_Detection}

		# print("Target Information: ", self.targets)
		# print("Self Information: ", self.states)
		# print("Neighbor Information: ", self.neighbors)

	# def qp_ini(self):
		
	# 	self.m = gp.Model("qp")
	# 	self.m.setParam("NonConvex", 2.0)
	# 	self.m.setParam("LogToConsole",0)
	# 	self.x = self.m.addVars(2,ub=0.5, lb=-0.5, name="x")

	# def addCons(self, i):

	# 	self.m.addConstr(self.A[i,0]*self.x[0] + self.A[i,1]*self.x[1] <= self.b[i], "c"+str(i))

	def	controller(self, dx, dp, step):

		'''
		P1, P2, P3, P4, P4v, PO = self.P1, self.P2, self.P3, self.P4, self.P4v, self.PO

		u_des = np.array([1*( (P2[0] - P3[0]) + 1.0 + (P1[0] - P3[0]) - 1.0 + 1*((P4[0] - P3[0]) + 0) + P4v[0] ),\
		1*( (P2[1] - P3[1]) + sqrt(3) + (P1[1] - P3[1]) + sqrt(3) + 1*((P4[1] - P3[1]) + 0.5*sqrt(3)) + P4v[1] ),\
		0.7 - P3[2]])

		obj = (self.x[0] - u_des[0])**2 + (self.x[1] - u_des[1])**2
		self.m.setObjective(obj)
		self.m.remove(self.m.getConstrs())

		for i in range (self.b.size):

			self.addCons(i)

		self.m.optimize()
		u_opt = self.m.getVars()

		self.cmd_vel.linear.x = u_opt[0].X
		self.cmd_vel.linear.y = u_opt[1].X
		self.cmd_vel.linear.z = u_des[2]
		'''

		self.cmd_vel.linear.x = 1.5*dx[0]*step
		self.cmd_vel.linear.y = 1.5*dx[1]*step
		self.cmd_vel.linear.z = 11.0 - self.P2[2]

		persp_t = self.perspective + dp*step
		persp_t /= np.linalg.norm(persp_t)

		axis = np.cross(self.perspective, persp_t)
		axis = axis/np.linalg.norm(axis)
		dot_product = np.dot(self.perspective, persp_t)
		dtheta = np.arccos( dot_product/(np.linalg.norm(self.perspective) * np.linalg.norm(persp_t)) )

		self.cmd_vel.angular.z = 20.0*dtheta*axis

		self.px4.vel_control(self.cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_2')
		rate = rospy.Rate(100)

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		camera2 = { 'id'            :  2,
					'ecode'         :  "/uav2",
					'position'      :  np.array([2.0, 23.0]),
					'perspective'   :  np.array([1.0, -1.0]),
					'AngleofView'   :  20,
					'range_limit'   :  4.0,
					'lambda'        :  2,
					'color'         : (0, 0, 200)}

		ptz_2 = PTZcon(camera2, map_size, grid_size)
		uav_2 = UAV(camera2["id"])
		# est_2 = DepthEstimator(camera2['ecode'], "/home/leo/mts/src/coverage/record/uav2.csv",
		# 				uav_2.targetName, uav_2.pixel_size, uav_2.image_width, uav_2.FL_Curr)

		# filename = "/home/leo/mts/src/coverage/record/uav2.csv"
		# f = open(filename, "w+")
		# f.close()

		while (uav_2.P2 is None) or (uav_2.neighbor_index is None):

			rate.sleep()

		# uav_2.qp_ini()

		last = time()

		while not rospy.is_shutdown():

			past = time()
			uav_2.Collect_Data()
			dx, dp, step = ptz_2.UpdateState(uav_2.targets, uav_2.neighbors, uav_2.states,\
												np.round(time() - last, 2))
			uav_2.controller(dx, dp, step)
			# est_2.MultiTarget_Estimation(uav_2.states, uav_2.neighbors)
			# uav_2.SetCameraZoom(1)

			print("Calculation Time 2: " + str(time() - past))

			rate.sleep()

	except rospy.ROSInterruptException:
		pass
