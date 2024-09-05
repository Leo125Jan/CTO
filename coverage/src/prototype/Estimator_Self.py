import csv
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo
from scipy.optimize import linear_sum_assignment
from coverage.msg import BoundingBox, BoundingBoxes
from gazebo_msgs.msg import ModelStates, LinkStates
from std_msgs.msg import Float64, Float64MultiArray


class DepthEstimator():

	def __init__(self, uav, filename, targetName, pixel_size, image_width, FL_Curr):

		# Parameter
		self.uav = uav
		self.targetName = targetName
		self.pixel_size = pixel_size
		self.image_width = image_width
		self.FL_Curr = FL_Curr
		self.filename = filename
		self.X = np.array([7.0, 7.0, 10.0])

		# ROS Publisher & Subscriber
		self.states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.States_callback, queue_size = 100)

		self.Essential_pub = rospy.Publisher(self.uav + "/Essential", Float64MultiArray, queue_size = 100)
		self.Detection_pub = rospy.Publisher(self.uav + "/Detection", Float64MultiArray, queue_size = 100)

		# Variable
		self.img = np.array([[0.0,0.0,0.0]])
		self.pixel = np.array([[0.0,0.0,0.0]])
		self.camerainfo = np.array([[0.0,0.0,0.0,0,0]])
		self.CamPos = None
		self.CamO = np.array([0.0,0.0,0.0,0.0])
		self.one_hop_neighbor = None

		self.P5, self.P6, self.P7, self.P8, self.P9 = None, None, None, None, None

	def States_callback(self, msg):

		UAVs_index = [msg.name.index('uav5'), msg.name.index('uav6'), msg.name.index('uav7'), msg.name.index('uav8'), msg.name.index('uav9')]

		P5 = np.array([msg.pose[UAVs_index[0]].position.x, msg.pose[UAVs_index[0]].position.y, msg.pose[UAVs_index[0]].position.z])
		P6 = np.array([msg.pose[UAVs_index[1]].position.x, msg.pose[UAVs_index[1]].position.y, msg.pose[UAVs_index[1]].position.z])
		P7 = np.array([msg.pose[UAVs_index[2]].position.x, msg.pose[UAVs_index[2]].position.y, msg.pose[UAVs_index[2]].position.z])
		P8 = np.array([msg.pose[UAVs_index[3]].position.x, msg.pose[UAVs_index[3]].position.y, msg.pose[UAVs_index[3]].position.z])
		P9 = np.array([msg.pose[UAVs_index[4]].position.x, msg.pose[UAVs_index[4]].position.y, msg.pose[UAVs_index[4]].position.z])
		
		self.P5, self.P6, self.P7, self.P8, self.P9 = P5, P6, P7, P8, P9
		self.target = np.array([self.P5, self.P6, self.P7, self.P8, self.P9])

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

	def MultiTarget_Estimation(self, States, Neighbors):

		self.img = States["img"]
		self.pixel = States["pixel"]
		self.camerainfo = States["camerainfo"]
		self.CamPos = States["CamPos"]
		self.CamO = States["CamOrientation"]
		self.one_hop_neighbor = States["NeighorIndex"]

		output_ = Float64MultiArray(data = self.CamO.tolist())
		# print("Essential: ", output_)
		self.Essential_pub.publish(output_)

		hold = []
		for element in self.img:

			hold.extend(element.tolist())
		output_ = Float64MultiArray(data = hold)
		# print("Detection: ", output_)
		self.Detection_pub.publish(output_)
		# print("Image Empty: ", np.array_equal(self.img, np.array([[0.0,0.0,0.0]])))
		# print("Pixel Empty: ", np.array_equal(self.pixel, np.array([[0.0,0.0,0.0]])))
		# print("Camerainfo Empty: ", np.array_equal(self.camerainfo, np.array([[0.0,0.0,0.0,0.0]])))
		
		# if False:
		if np.array_equal(self.img, np.array([[0.0,0.0,0.0]])) or\
			np.array_equal(self.pixel, np.array([[0.0,0.0,0.0]])) or\
			np.array_equal(self.camerainfo, np.array([[0.0,0.0,0.0,0.0]])):

			pass
		else:
			Pass = True
			for key in self.one_hop_neighbor:

				# print("Neighbor Detection: ", Neighbors["Detection"][str(key)])

				if np.array_equal(Neighbors["Detection"][str(key)], np.array([[0.0,0.0,0.0]])):

					Pass = False
				if (np.array(Neighbors["Detection"][str(key)]) == None).all():

					Pass = False
				if (np.array(Neighbors["Essential"][str(key)]) == None).all():

					Pass = False

				# print("Pass: ", Pass)

			if not Pass:

				pass
			else:

				Rx = np.array([[1, 0, 0],
							[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
							[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
				Rz = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi), 0],
							[np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi), 0],
							[0, 0, 1]])

				tau_self = []
				for img in States["img"]:

					tau = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
					tau_self.append(tau)
				tau_self = np.array(tau_self)
				# print("tau_self: ", tau_self)

				tau_other = {str(neighbor_id): [] for neighbor_id in self.one_hop_neighbor}

				for key in self.one_hop_neighbor:

					for img in Neighbors["Detection"][str(key)]:

						# print("tau: ", tau)
						tau_hold = np.matmul(Rz, np.matmul(Rx, np.reshape(img, (3,1))))
						tau_other[str(key)].append(tau_hold)
					tau_other[str(key)] = np.array(tau_other[str(key)])

				# print("tau_other: ", tau_other)

				# # Rotation matrices from body frame to interial
				R_self = np.array(self.q2R(self.CamO))
				R_other = {str(neighbor_id): None for neighbor_id in self.one_hop_neighbor}
				for key in self.one_hop_neighbor:

					R = np.array(self.q2R(Neighbors["Essential"][str(key)]))
					R_other[str(key)] = R
				# print("R_self: ", R_self)
				# print("R_other: ", R_other)

				# Essential Matrix
				E = {str(neighbor_id): None for neighbor_id in self.one_hop_neighbor}
				for key in self.one_hop_neighbor:

					baseline = self.CamPos - Neighbors["CamPos"][str(key)]; baseline /= np.linalg.norm(baseline)
					skew_matrix = np.array([[0, -baseline[2], baseline[1]],
			                     [baseline[2], 0, -baseline[0]],
			                     [-baseline[1], baseline[0], 0]])
					# print("R0 T: ", np.transpose(R0))
					E[str(key)] = np.matmul(np.transpose(R_self), np.matmul(skew_matrix, R_other[str(key)]))
					# print("E: ", E)

				C_M = []
				for index, tau in tau_other.items():

					C = []

					for tau_A in tau_self:

						C_ = [abs(np.matmul(np.transpose(tau_A), np.matmul(E[str(index)], tau_B))[0])[0] for tau_B in tau]
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
				for i, tau_A in enumerate(tau_self):

					hold = []; hold.append(tau_A)

					for j, key in enumerate(tau_other):

						hold.append(tau_other[str(key)][col[j][i]])

					detection.append(hold)


				# detection = []

				# for tau_A in tau_self:

				# 	hold = []; hold.append(tau_A)

				# 	for key in tau_other:

				# 		min_value = np.inf
				# 		index = None

				# 		# print("tau_A: ", tau_A)
				# 		# print("tau_B: ", np.matmul(np.transpose(tau_A), np.matmul(E, tau_B)))

				# 		for i, tau in enumerate(tau_other[key]):

				# 			value = abs(np.matmul(np.transpose(tau_A), np.matmul(E[key], tau))[0])
				# 			# print("value: ", value)

				# 			if value < min_value:

				# 				min_value = value
				# 				index = i
				# 		# print("index: ", index)
				# 		# print(halt)

				# 		hold.append(tau_other[key][index])
				# 	detection.append(hold)

				# # print("detection: ", detection)

				Pt = []
				for pair in detection:

					Pt.append(self.Main(pair, Neighbors))

				print(self.uav + " Pt: ", Pt)

				# target = self.target[States["ClusterSet"]]
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
					data.extend(element_)
					data.extend(Pt[index])

				# filename = self.filename
				# with open(filename, "a", encoding='UTF8', newline='') as f:

				# 	row = data
				# 	writer = csv.writer(f)
				# 	writer.writerow(row)


	def Main(self, tau, Neighbors):

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
		R = []
		R_self = np.array(self.q2R(self.CamO)); R.append(R_self)
		
		for key in self.one_hop_neighbor:

			hold = np.array(self.q2R(Neighbors["Essential"][str(key)]))
			R.append(hold)
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
		Cam = []; Cam.append(self.CamPos)
		for key in self.one_hop_neighbor:

			pos = np.array(Neighbors["CamPos"][str(key)])
			Cam.append(pos)

		B = np.array([])
		for i in range(n):

			if i == 0:

				B = -np.reshape(Cam[i], (3,1))
			else:

				B = np.vstack((B, -np.reshape(Cam[i], (3,1))))
		# print("B: ", B)
		# print("B shape: ", np.shape(B))

		# Solution - X
		# print("A: ", A)
		# print("A.T: ", A.T)

		try:

			X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B))
			# print("X: ", X)
			Pt = -np.reshape(X[n:len(X)], (1,3))[0]
			self.X = X
		except:
			Pt = -np.reshape(self.X[n:len(self.X)], (1,3))[0]
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