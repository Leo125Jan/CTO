import csv
import rospy
import numpy as np
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo
from scipy.optimize import linear_sum_assignment
from coverage.msg import BoundingBox, BoundingBoxes
from gazebo_msgs.msg import ModelStates, LinkStates


class DepthEstimator():

	def __init__(self, uav, filename):

		# ROS Publisher & Subscriber
		self.uav1_hfov_pub = rospy.Publisher("/uav1/set_zoom", Float64, queue_size=100)
		self.uav2_hfov_pub = rospy.Publisher("/uav2/set_zoom", Float64, queue_size=100)
		self.uav3_hfov_pub = rospy.Publisher("/uav3/set_zoom", Float64, queue_size=100)
		self.targetTruth_pub = rospy.Publisher("/TargetGroundTruth", Pose, queue_size=100)
		self.targetEstimation_pub = rospy.Publisher("/TargetEstimation", Pose, queue_size=100)

		self.link_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.Link_callback, queue_size = 100)
		self.states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.States_callback, queue_size = 100)

		self.uav1_camerainfo_sub = rospy.Subscriber("/uav1/camera/camera/color/camera_info", CameraInfo, self.CameraInfo1_callback, queue_size = 100)
		self.uav2_camerainfo_sub = rospy.Subscriber("/uav2/camera/camera/color/camera_info", CameraInfo, self.CameraInfo2_callback, queue_size = 100)
		self.uav3_camerainfo_sub = rospy.Subscriber("/uav3/camera/camera/color/camera_info", CameraInfo, self.CameraInfo3_callback, queue_size = 100)
		self.uav1_recognition_sub = rospy.Subscriber("/uav1/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition1_callback, queue_size = 100)
		self.uav2_recognition_sub = rospy.Subscriber("/uav2/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition2_callback, queue_size = 100)
		self.uav3_recognition_sub = rospy.Subscriber("/uav3/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition3_callback, queue_size = 100)

		# Parameter
		self.uav = uav
		self.pixel_size = 2.9*1e-6
		self.image_width = 1280
		self.FL_1x = 1280.785 # 6.5mm/2.9um
		self.FL_Curr = 1280.785
		self.filename = filename

		# Variable
		self.targetName = "Drone"
		self.img1, self.img2, self.img3 = None, None, None
		self.pixel1, self.pixel2, self.pixel3 = None, None, None
		self.camerainfo1, self.camerainfo2, self.camerainfo3 = None, None, None
		self.P1, self.P1R, self.P2, self.P2R, self.P3, self.P3R = None, None, None, None, None, None
		self.Cam1, self.Cam1R, self.Cam2, self.Cam2R, self.Cam3, self.Cam3R = None, None, None, None, None, None

	def Link_callback(self, msg):

		Link_index = [msg.name.index('uav1::cgo3_camera_link'), msg.name.index('uav2::cgo3_camera_link'),\
																msg.name.index('uav3::cgo3_camera_link')]

		Cam1 = np.array([msg.pose[Link_index[0]].position.x, msg.pose[Link_index[0]].position.y, msg.pose[Link_index[0]].position.z])
		Cam1o = np.array([msg.pose[Link_index[0]].orientation.x, msg.pose[Link_index[0]].orientation.y,\
						msg.pose[Link_index[0]].orientation.z, msg.pose[Link_index[0]].orientation.w])

		Cam2 = np.array([msg.pose[Link_index[1]].position.x, msg.pose[Link_index[1]].position.y, msg.pose[Link_index[1]].position.z])
		Cam2o = np.array([msg.pose[Link_index[1]].orientation.x, msg.pose[Link_index[1]].orientation.y,\
						msg.pose[Link_index[1]].orientation.z, msg.pose[Link_index[1]].orientation.w])

		Cam3 = np.array([msg.pose[Link_index[1]].position.x, msg.pose[Link_index[1]].position.y, msg.pose[Link_index[1]].position.z])
		Cam3o = np.array([msg.pose[Link_index[1]].orientation.x, msg.pose[Link_index[1]].orientation.y,\
						msg.pose[Link_index[1]].orientation.z, msg.pose[Link_index[1]].orientation.w])

		self.Cam1, self.Cam2, self.Cam3 = Cam1, Cam2, Cam3
		self.Cam1R, self.Cam2R, self.Cam3R = self.q2R(Cam1o), self.q2R(Cam2o), self.q2R(Cam3o)
		self.Cam = np.array([self.Cam1, self.Cam2, self.Cam3])
		# print("Cam: ", self.Cam)

	def States_callback(self, msg):

		UAV_index = [msg.name.index('uav1'), msg.name.index('uav2'), msg.name.index('uav3'),\
					msg.name.index('uav7'), msg.name.index('uav8'), msg.name.index('uav9')]

		P1 = np.array([msg.pose[UAV_index[0]].position.x, msg.pose[UAV_index[0]].position.y, msg.pose[UAV_index[0]].position.z])
		P1o = np.array([msg.pose[UAV_index[0]].orientation.x, msg.pose[UAV_index[0]].orientation.y,\
						msg.pose[UAV_index[0]].orientation.z, msg.pose[UAV_index[0]].orientation.w])

		P2 = np.array([msg.pose[UAV_index[1]].position.x, msg.pose[UAV_index[1]].position.y, msg.pose[UAV_index[1]].position.z])
		P2o = np.array([msg.pose[UAV_index[1]].orientation.x, msg.pose[UAV_index[1]].orientation.y,\
						msg.pose[UAV_index[1]].orientation.z, msg.pose[UAV_index[1]].orientation.w])

		P3 = np.array([msg.pose[UAV_index[2]].position.x, msg.pose[UAV_index[2]].position.y, msg.pose[UAV_index[2]].position.z])
		P3o = np.array([msg.pose[UAV_index[2]].orientation.x, msg.pose[UAV_index[2]].orientation.y,\
						msg.pose[UAV_index[2]].orientation.z, msg.pose[UAV_index[2]].orientation.w])

		P7 = np.array([msg.pose[UAV_index[3]].position.x, msg.pose[UAV_index[3]].position.y, msg.pose[UAV_index[3]].position.z])
		P8 = np.array([msg.pose[UAV_index[4]].position.x, msg.pose[UAV_index[4]].position.y, msg.pose[UAV_index[4]].position.z])
		P9 = np.array([msg.pose[UAV_index[5]].position.x, msg.pose[UAV_index[5]].position.y, msg.pose[UAV_index[5]].position.z])


		self.P1, self.P2, self.P3, self.P7, self.P8, self.P9 = P1, P2, P3, P7, P8, P9
		self.P1R, self.P2R, self.P3R = self.q2R(P1o), self.q2R(P2o), self.q2R(P3o)
		self.P = np.array([self.P1, self.P2, self.P3])
		# print("P: ", self.P)

	def q2rpy(self, q):

		if isinstance(q, Quaternion):

			rotate_x_rad = q.yaw_pitch_roll[2]
			rotate_y_rad = q.yaw_pitch_roll[1]
			rotate_z_rad = q.yaw_pitch_roll[0]
		else:

			q_ = Quaternion(q[3], q[0], q[1], q[2])
			rotate_x_rad = q_.yaw_pitch_roll[2]
			rotate_y_rad = q_.yaw_pitch_roll[1]
			rotate_z_rad = q_.yaw_pitch_roll[0]

		return np.array([rotate_x_rad, rotate_y_rad, rotate_z_rad])

	def q2R(self, q):

		if isinstance(q, Quaternion):

			R = q.rotation_matrix 
		else:

			q_ = Quaternion(q[3], q[0], q[1], q[2])
			R = q_.rotation_matrix 

		return R

	def CameraInfo1_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo1 = np.array([fx, fy, cx, cy])

	def CameraInfo2_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo2 = np.array([fx, fy, cx, cy])

	def CameraInfo3_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo3 = np.array([fx, fy, cx, cy])

	def Recognition1_callback(self, msg):

		pixel1, img1 = [], []

		for element in msg.bounding_boxes:

			if element.Class == self.targetName:

				xmin = element.xmin
				xmax = element.xmax
				ymin = element.ymin
				ymax = element.ymax

				# Pixel frame
				u = (xmin+xmax)/2
				v = (ymin+ymax)/2
				pixel1.append(np.array([u ,v]))
				# print("UAV0 - u&v: ", u, v)

				# Pixel frame convert to image plane (in meter)
				img1_hold = np.array([(u-self.camerainfo1[2]), (v-self.camerainfo1[3]), (self.FL_Curr)])
				# print("img1: ", self.img1)
				img1_hold /= np.linalg.norm(img1_hold)
				# print("img1 unit: ", self.img1)
				img1.append(img1_hold)

		if len(pixel1) == 3:

			self.pixel1 = pixel1
			self.img1 = img1
		elif len(pixel1) < 3 and self.pixel1 != None and self.img1 != None:

			check_list = list(range(3))

			for point_B in pixel1:

				min_distance = float('inf')
				closest_index = None

				for i, point_A in enumerate(self.pixel1):

					distance = np.linalg.norm(point_B-point_A)

					if distance < min_distance:

						min_distance = distance
						closest_index = i

				check_list.remove(closest_index)

			for index in check_list:

				pixel1.append(self.pixel1[index])
				img1.append(self.img1[index])

			self.pixel1 = pixel1
			self.img1 = img1

		# print("pixel1: ", self.pixel1)
		# print("img1: ", self.img1)

	def Recognition2_callback(self, msg):

		pixel2, img2 = [], []

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

		if len(pixel2) == 3:

			self.pixel2 = pixel2
			self.img2 = img2
		elif len(pixel2) < 3 and self.pixel2 != None and self.img2 != None:

			check_list = list(range(3))

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

			self.pixel2 = pixel2
			self.img2 = img2

	def Recognition3_callback(self, msg):

		pixel3, img3 = [], []

		for element in msg.bounding_boxes:

			if element.Class == self.targetName:

				xmin = element.xmin
				xmax = element.xmax
				ymin = element.ymin
				ymax = element.ymax

				# Pixel frame
				u = (xmin+xmax)/2
				v = (ymin+ymax)/2
				pixel3.append(np.array([u ,v]))
				# print("UAV0 - u&v: ", u, v)

				# Pixel frame convert to image plane (in meter)
				img3_hold = np.array([(u-self.camerainfo3[2]), (v-self.camerainfo3[3]), (self.FL_Curr)])
				# print("img0: ", self.img0)
				img3_hold /= np.linalg.norm(img3_hold)
				# print("img0 unit: ", self.img0)
				img3.append(img3_hold)

		if len(pixel3) == 3:

			self.pixel3 = pixel3
			self.img3 = img3
		elif len(pixel3) < 3 and self.pixel3 != None and self.img3 != None:

			check_list = list(range(3))

			for point_B in pixel3:
				min_distance = float('inf')
				closest_index = None

				for i, point_A in enumerate(self.pixel3):

					distance = np.linalg.norm(point_B-point_A)

					if distance < min_distance:

						min_distance = distance
						closest_index = i

				check_list.remove(closest_index)

			for index in check_list:

				pixel3.append(self.pixel3[index])
				img3.append(self.img3[index])

			self.pixel3 = pixel3
			self.img3 = img3

	def SetCameraZoom(self, zoom):

		output = Float64(data = zoom)

		self.uav1_hfov_pub.publish(output)
		self.uav2_hfov_pub.publish(output)
		self.uav3_hfov_pub.publish(output)

		self.FL_Curr = zoom*self.FL_1x

	def MultiTarget_Estimation(self):

		Rx = np.array([[1, 0, 0],
					[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
					[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
		Rz = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi), 0],
					[np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi), 0],
					[0, 0, 1]])

		tau1, tau2, tau3 = [], [], []
		for img in self.img1:

			tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
			tau1.append(tau)
		tau1 = np.array(tau1)

		for img in self.img2:

			tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
			tau2.append(tau)
		tau2 = np.array(tau2)

		for img in self.img3:

			tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
			tau3.append(tau)
		tau3 = np.array(tau3)

		# print("pixel0: ", self.pixel0)
		# print("tau0: ", tau0)
		# print("pixel4: ", self.pixel4)
		# print("tau4: ", tau4)

		# # Rotation matrices from body frame to interial
		R1 = np.array(self.Cam1R)
		R2 = np.array(self.Cam2R)
		R3 = np.array(self.Cam3R)
		# print("R0: ", R0)

		# Essential Matrix
		E = []

		baseline = self.Cam1 - self.Cam2; baseline /= np.linalg.norm(baseline)
		skew_matrix = np.array([[0, -baseline[2], baseline[1]],
                     [baseline[2], 0, -baseline[0]],
                     [-baseline[1], baseline[0], 0]])
		# print("R0 T: ", np.transpose(R0))
		E_ = np.matmul(np.transpose(R1), np.matmul(skew_matrix, R2))
		E.append(E_)
		# print("E: ", E)

		baseline = self.Cam1 - self.Cam3; baseline /= np.linalg.norm(baseline)
		skew_matrix = np.array([[0, -baseline[2], baseline[1]],
                     [baseline[2], 0, -baseline[0]],
                     [-baseline[1], baseline[0], 0]])
		# print("R0 T: ", np.transpose(R0))
		E_ = np.matmul(np.transpose(R1), np.matmul(skew_matrix, R3))
		E.append(E_)

		C_M = []
		for i, tau in enumerate(np.array([tau2, tau3])):

			C = []

			for tau_A in tau1:

				C_ = [abs(np.matmul(np.transpose(tau_A), np.matmul(E[i], tau_B))[0])[0] for tau_B in tau]
				C.append(C_)

			# print("C: ", C)
			C_M.append(C)
			# print("C_M: ", C_M)
		C_M = np.array(C_M)
		print("C_M: ", C_M)

		row, col = [], []

		for cost_matrix in C_M:

			print("cost_matrix: ", cost_matrix)
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			row.append(row_ind)
			col.append(col_ind)
		print("row: ", row)
		print("col: ", col)

		detection = []
		for i, tau_A in enumerate(tau1):

			hold = []; hold.append(tau_A)

			for j, tau in enumerate(np.array([tau2, tau3])):

				hold.append(tau[col[j][i]])

			detection.append(hold)
		# print("detection: ", detection)

		# detection = []

		# for tau_A in tau1:

		# 	hold = []; hold.append(tau_A)

		# 	for j, tau in enumerate(np.array([tau2, tau3])):

		# 		min_value = np.inf
		# 		index = None

		# 		for i, tau_B in enumerate(tau):

		# 			print("i: ", i)
		# 			# print("tau_A: ", tau_A)
		# 			# print("value: ", abs(np.matmul(np.transpose(tau_A), np.matmul(E[j], tau_B))[0]))

		# 			value = abs(np.matmul(np.transpose(tau_A), np.matmul(E[j], tau_B))[0])
		# 			print("value: ", value)

		# 			if value < min_value:

		# 				min_value = value
		# 				index = i
		# 		print("index: ", index)

		# 		hold.append(tau[index])
		# 	detection.append(hold)
		# 	print("\n")

		# print("detection: ", detection)

		Pt = []
		for pair in detection:

			Pt.append(self.Main(pair))

		target = np.array([self.P7, self.P8, self.P9])
		# print("target: ", target)
		data = []

		for element_ in target:

			index = None
			min_ = np.inf

			for i, element in enumerate(Pt):

				value = np.linalg.norm(element-element_)

				if value < min_:

					min_ = value
					index = i

			print("Pt E: ", element_)
			print("Pt T: ", Pt[index])
			print("Pt Error: ", np.linalg.norm(element_-Pt[index]))

	def Main(self, tau):

		n = len(tau)
		# Rx = np.array([[1, 0, 0],
		# 			[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
		# 			[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
		# Rz = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi), 0],
		# 			[np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi), 0],
		# 			[0, 0, 1]])

		# Rotate from image plane(camera frame) to body frame
		# tau0 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img0, (3,1))))
		# tau1 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img1, (3,1))))
		# tau2 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img2, (3,1))))
		# tau = [tau0, tau1, tau2]
		# print("tau: ", tau)
		# print("tau0: ", tau0)

		# Rotation matrices from body frame to interial
		R1 = self.Cam1R
		R2 = self.Cam2R
		R3 = self.Cam3R
		R = [R1, R2, R3]
		# print("R: ", R)
		# print("result: ", np.allclose(np.matmul(R0, self.P0rpw), np.eye(3)))

		# Optimization - A
		A = np.array([])
		for i in range(n):

			hold = np.array([])

			for j in range(n):

				if j == i and i == 0:

					hold = np.matmul(R[i], tau[i])
				elif j == i and i != 0:

					hold = np.vstack((hold, np.matmul(R[i], tau[i])))
				elif j == 0 and i != 0:

					hold = np.zeros((3,1))
				else:

					hold = np.vstack((hold, np.zeros((3,1))))
				# print("hold: ", hold)

			if i == 0:
				
				A = hold
			else:

				A = np.hstack((A, hold))
			# print("A: ", A)

		I = np.array([])
		for i in range(int(np.shape(A)[0]/3)):

			if i == 0:

				I = np.identity(3)
			else:

				I = np.vstack((I, np.identity(3)))
		# print("I: ", I)

		A = np.hstack((A, I))
		# print("Cost A: ", A)
		# print("A shape: ", np.shape(A))

		# Optimization - B
		B = np.array([])
		for i in range(n):

			if i == 0:

				B = -np.reshape(self.Cam[i], (3,1))
			else:

				B = np.vstack((B, -np.reshape(self.Cam[i], (3,1))))
		# print("B: ", B)
		# print("B shape: ", np.shape(B))

		# Solution - X
		# print("A: ", A)
		# print("A.T: ", A.T)
		X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B))
		# print("X: ", X)
		Pt = -np.reshape(X[n:len(X)], (1,3))[0]
		# print("Pt: ", Pt)
		# print("P5: ", self.P5)
		# print("P6: ", self.P6)
		# print("Error: ", np.linalg.norm(Pt-self.wamv))

		# output_ = Pose()
		# output_.position.x = self.wamv[0]
		# output_.position.y = self.wamv[1]
		# output_.position.z = self.wamv[2]
		# self.targetTruth_pub.publish(output_)

		# output_ = Pose()
		# output_.position.x = Pt[0]
		# output_.position.y = Pt[1]
		# output_.position.z = Pt[2]
		# self.targetEstimation_pub.publish(output_)

		# filename = self.filename
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = [self.wamv[0], self.wamv[1], self.wamv[2], Pt[0], Pt[1], Pt[2]]
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

		return Pt


if __name__ == '__main__':

	rospy.init_node('Estimator', anonymous=True)

	filename = "/home/leo/plan/src/model_test/data/Data_MoveWamv_Record.csv"
	# f = open(filename, "w+")
	# f.close()

	node = DepthEstimator(3, filename)

	rate = rospy.Rate(100)

	while (node.P1 is None) or (node.P2 is None) or (node.P3 is None) or\
			(node.Cam1 is None) or (node.Cam2 is None) or (node.Cam3 is None) or\
			(node.camerainfo1 is None) or (node.camerainfo2 is None) or (node.camerainfo3 is None) or\
			(node.img1 is None) or (node.img2 is None) or (node.img3 is None) or\
			(node.pixel1 is None) or (node.pixel2 is None) or (node.pixel3 is None):

			rate = rospy.Rate(100)

	rospy.loginfo("Estimator StandBy")

	while not rospy.is_shutdown():

		node.SetCameraZoom(1)
		# node.Main()
		node.MultiTarget_Estimation()

		rate.sleep()