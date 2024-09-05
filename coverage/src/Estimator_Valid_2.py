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
		self.uav0_hfov_pub = rospy.Publisher("/uav0/set_zoom", Float64, queue_size=100)
		self.uav4_hfov_pub = rospy.Publisher("/uav4/set_zoom", Float64, queue_size=100)
		self.targetTruth_pub = rospy.Publisher("/TargetGroundTruth", Pose, queue_size=100)
		self.targetEstimation_pub = rospy.Publisher("/TargetEstimation", Pose, queue_size=100)

		self.link_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self.Link_callback, queue_size = 100)
		self.states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.States_callback, queue_size = 100)

		self.uav0_camerainfo_sub = rospy.Subscriber("/uav0/camera/camera/color/camera_info", CameraInfo, self.CameraInfo0_callback, queue_size = 100)
		self.uav4_camerainfo_sub = rospy.Subscriber("/uav4/camera/camera/color/camera_info", CameraInfo, self.CameraInfo4_callback, queue_size = 100)
		self.uav0_recognition_sub = rospy.Subscriber("/uav0/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition0_callback, queue_size = 100)
		self.uav4_recognition_sub = rospy.Subscriber("/uav4/yolov8/BoundingBoxes", BoundingBoxes, self.Recognition4_callback, queue_size = 100)

		# Parameter
		self.uav = uav
		self.pixel_size = 2.9*1e-6
		self.image_width = 1280
		self.FL_1x = 1280.785 # 6.5mm/2.9um
		self.FL_Curr = 1280.785
		self.filename = filename

		# Variable
		self.targetName = "Drone"
		self.img0, self.img4 = None, None
		self.pixel0, self.pixel4 = None, None
		self.camerainfo0, self.camerainfo4 = None, None
		self.P0, self.P0R, self.P4, self.P4R = None, None, None, None
		self.Cam0, self.Cam0R, self.Cam0rpy, self.Cam4, self.Cam4R, self.Cam4rpy = None, None, None, None, None, None

	def Link_callback(self, msg):

		Link_index = [msg.name.index('uav0::cgo3_camera_link'), msg.name.index('uav4::cgo3_camera_link')]

		Cam0 = np.array([msg.pose[Link_index[0]].position.x, msg.pose[Link_index[0]].position.y, msg.pose[Link_index[0]].position.z])
		Cam0o = np.array([msg.pose[Link_index[0]].orientation.x, msg.pose[Link_index[0]].orientation.y,\
						msg.pose[Link_index[0]].orientation.z, msg.pose[Link_index[0]].orientation.w])

		Cam4 = np.array([msg.pose[Link_index[1]].position.x, msg.pose[Link_index[1]].position.y, msg.pose[Link_index[1]].position.z])
		Cam4o = np.array([msg.pose[Link_index[1]].orientation.x, msg.pose[Link_index[1]].orientation.y,\
						msg.pose[Link_index[1]].orientation.z, msg.pose[Link_index[1]].orientation.w])

		self.Cam0, self.Cam4 = Cam0, Cam4
		self.Cam0R, self.Cam4R = self.q2R(Cam0o), self.q2R(Cam4o)
		self.Cam0rpy, self.Cam4rpy = self.q2rpy(Cam0o), self.q2rpy(Cam4o)
		self.Cam = np.array([self.Cam0, self.Cam4])
		# print("Cam0R: ", self.Cam0R)
		# print("Cam0rpy: ", self.q2rpy(Cam0o))
		# print("Cam4rpy: ", self.q2rpy(Cam4o))

	def States_callback(self, msg):

		UAV_index = [msg.name.index('uav0'), msg.name.index('uav4'), msg.name.index('uav5'), msg.name.index('uav6')]

		P0 = np.array([msg.pose[UAV_index[0]].position.x, msg.pose[UAV_index[0]].position.y, msg.pose[UAV_index[0]].position.z])
		P0o = np.array([msg.pose[UAV_index[0]].orientation.x, msg.pose[UAV_index[0]].orientation.y,\
						msg.pose[UAV_index[0]].orientation.z, msg.pose[UAV_index[0]].orientation.w])

		P4 = np.array([msg.pose[UAV_index[1]].position.x, msg.pose[UAV_index[1]].position.y, msg.pose[UAV_index[1]].position.z])
		P4o = np.array([msg.pose[UAV_index[1]].orientation.x, msg.pose[UAV_index[1]].orientation.y,\
						msg.pose[UAV_index[1]].orientation.z, msg.pose[UAV_index[1]].orientation.w])

		P5 = np.array([msg.pose[UAV_index[2]].position.x, msg.pose[UAV_index[2]].position.y, msg.pose[UAV_index[2]].position.z])
		P6 = np.array([msg.pose[UAV_index[3]].position.x, msg.pose[UAV_index[3]].position.y, msg.pose[UAV_index[3]].position.z])


		self.P0, self.P4, self.P5, self.P6 = P0, P4, P5, P6
		self.P0R, self.P4R = self.q2R(P0o), self.q2R(P4o) 
		self.P = np.array([self.P0, self.P4])
		# print("P0R: ", self.P0R)
		# print("P0rpy: ", self.q2rpy(P0o))

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

	def CameraInfo0_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo0 = np.array([fx, fy, cx, cy])
		self.K1 = np.reshape(msg.K, (3,3))
		# print("K1: ", self.K1)

	def CameraInfo4_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo4 = np.array([fx, fy, cx, cy])
		self.K4 = np.reshape(msg.K, (3,3))
		# print("K4: ", self.K4)

	def Recognition0_callback(self, msg):

		pixel0, img0 = [], []

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
				# img0_hold /= np.linalg.norm(img0_hold)
				# print("img0 unit: ", self.img0)
				img0.append(img0_hold)

		if len(pixel0) == 2:

			self.pixel0 = pixel0
			self.img0 = img0
		elif len(pixel0) < 2 and self.pixel0 != None and self.img0 != None:

			check_list = list(range(2))

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

			self.pixel0 = pixel0
			self.img0 = img0

		# print("pixel0: ", self.pixel0)
		# print("img0: ", self.img0)

	def Recognition4_callback(self, msg):

		pixel4, img4 = [], []

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
				# img4_hold /= np.linalg.norm(img4_hold)
				# print("img0 unit: ", self.img0)
				img4.append(img4_hold)

		if len(pixel4) == 2:

			self.pixel4 = pixel4
			self.img4 = img4
		elif len(pixel4) < 2 and self.pixel4 != None and self.img4 != None:

			check_list = list(range(2))

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

			self.pixel4 = pixel4
			self.img4 = img4

	def SetCameraZoom(self, zoom):

		output = Float64(data = zoom)

		self.uav0_hfov_pub.publish(output)
		self.uav4_hfov_pub.publish(output)

		self.FL_Curr = zoom*self.FL_1x

	def MultiTarget_Estimation(self):

		Rx = np.array([[1, 0, 0],
					[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
					[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
		Rz = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi), 0],
					[np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi), 0],
					[0, 0, 1]])

		tau0, tau4 = [], []
		for img in self.img0:

			# Rotate tau from current camera frame to standard camera frame
			# Rpitch = np.array([[np.cos(-abs(self.Cam0rpy[1])), 0, np.sin(-abs(self.Cam0rpy[1]))],
			# 				[0, 1, 0],
			# 				[-np.sin(-abs(self.Cam0rpy[1])), 0, np.cos(-abs(self.Cam0rpy[1]))]])
			# img = np.matmul(Rpitch, np.reshape(img, (3,1)))

			tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
			tau0.append(tau)
		tau0 = np.array(tau0)

		for img in self.img4:

			# Rotate tau from current camera frame to standard camera frame
			# Rpitch = np.array([[np.cos(-abs(self.Cam4rpy[1])), 0, np.sin(-abs(self.Cam4rpy[1]))],
			# 				[0, 1, 0],
			# 				[-np.sin(-abs(self.Cam4rpy[1])), 0, np.cos(-abs(self.Cam4rpy[1]))]])
			# img = np.matmul(Rpitch, np.reshape(img, (3,1)))

			tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
			tau4.append(tau)
		tau4 = np.array(tau4)

		# print("pixel0: ", self.pixel0)
		# print("tau0: ", tau0)
		# print("pixel4: ", self.pixel4)
		# print("tau4: ", tau4)

		# # Rotation matrices from body frame to interial
		R0 = np.array(self.Cam0R)
		R4 = np.array(self.Cam4R)
		# print("R0: ", R0)

		# # Essential Matrix
		baseline = self.Cam0 - self.Cam4; baseline /= np.linalg.norm(baseline)
		skew_matrix = np.array([[0, -baseline[2], baseline[1]],
                     [baseline[2], 0, -baseline[0]],
                     [-baseline[1], baseline[0], 0]])
		# print("R0 T: ", np.transpose(R0))
		E = np.matmul(np.transpose(R0), np.matmul(skew_matrix, R4))
		# print("E: ", E)

		C_M = []
		for i, tau in enumerate(np.array([tau4])):

			C = []

			for tau_A in tau0:

				C_ = [abs(np.matmul(np.transpose(tau_A), np.matmul(E, tau_B))[0])[0] for tau_B in tau]
				C.append(C_)

			# print("C: ", C)
			C_M.append(C)
			# print("C_M: ", C_M)
		C_M = np.array(C_M)
		# print("C_M: ", C_M)

		row, col = [], []

		for cost_matrix in C_M:

			# print("cost_matrix: ", cost_matrix)
			row_ind, col_ind = linear_sum_assignment(cost_matrix)
			row.append(row_ind)
			col.append(col_ind)
		# print("row: ", row)
		# print("col: ", col)

		detection = []
		for i, tau_A in enumerate(tau0):

			hold = []; hold.append(tau_A)

			for j, tau in enumerate(np.array([tau4])):

				hold.append(tau[col[j][i]])

			detection.append(hold)
		# print("detection: ", detection)

		# detection = []

		# for tau_A in tau0:

		# 	hold = []; hold.append(tau_A)
		# 	min_value = np.inf
		# 	index = None

		# 	for i, tau_B in enumerate(tau4):

		# 		print("i: ", i)
		# 		# print("tau_A: ", tau_A)
		# 		# print("tau_B: ", np.matmul(np.transpose(tau_A), np.matmul(E, tau_B)))

		# 		value = abs(np.matmul(np.transpose(tau_A), np.matmul(E, tau_B))[0])
		# 		print("value: ", value)

		# 		if value < min_value:

		# 			min_value = value
		# 			index = i
		# 	# print("index: ", index)

		# 	hold.append(tau4[index])
		# 	detection.append(hold)
		# 	print("\n")

		# print("detection: ", detection)

		Pt = []
		for pair in detection:

			Pt.append(self.Main(pair))

		index = None
		min_ = np.inf

		for i, element in enumerate(Pt):

			value = np.linalg.norm(self.P5-element)

			if value < min_:

				min_ = value
				index = i

		# print("P5 E: ", Pt[index])
		# print("P5 T: ", self.P5)
		# print("P5 error: ", np.linalg.norm(Pt[index]-self.P5))

		index = None
		min_ = np.inf

		for i, element in enumerate(Pt):

			value = np.linalg.norm(self.P6-element)

			if value < min_:

				min_ = value
				index = i

		# print("P6 E: ", Pt[index])
		# print("P5 T: ", self.P6)
		# print("P6 error: ", np.linalg.norm(Pt[index]-self.P6))

		# Homographt Matrix
		R = np.matmul(np.linalg.inv(self.Cam0R), self.Cam4R)
		t = self.Cam4 - self.Cam0
		t_ = np.matmul(np.linalg.inv(self.Cam4R), np.reshape(t, (3,1)))

		

		# n = np.reshape([0,0,1], (3,1))
		# temp = np.subtract(np.eye(3), np.matmul(t, n))

		# H = np.matmul(R, temp)
		# print("H: ", H)

		# print("img4: ", self.img4)

		# for tau in tau4:
		# for img in self.img4:

			# rt = np.matmul(H, np.reshape(tau, (3,1)))
			
			# rt = np.matmul(self.K1, np.matmul(R, np.subtract(np.matmul(np.linalg.inv(self.K4), np.reshape(img, (3,1))), t_)))
			# rt /= np.linalg.norm(rt)
			# print("tau: ", rt)

		# print("img0: ", self.img0)
		# print("\n")


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
		R0 = self.Cam0R
		R4 = self.Cam4R
		R = [R0, R4]
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

	while (node.P0 is None) or (node.P4 is None) or\
			(node.Cam0 is None) or (node.Cam4 is None) or\
			(node.camerainfo0 is None) or (node.camerainfo4 is None) or\
			(node.img0 is None) or (node.img4 is None) or\
			(node.pixel0 is None) or (node.pixel4 is None):

			rate = rospy.Rate(100)

	rospy.loginfo("Estimator StandBy")

	while not rospy.is_shutdown():

		node.SetCameraZoom(1)
		# node.Main()
		node.MultiTarget_Estimation()

		rate.sleep()