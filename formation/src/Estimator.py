import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from pyquaternion import Quaternion
from sensor_msgs.msg import CameraInfo
from gazebo_msgs.msg import ModelStates
from darknet_ros_msgs.msg import BoundingBoxes

class DepthEstimator():

	def __init__(self, uav):

		# ROS Publisher & Subscriber
		self.uav0_hfov_pub = rospy.Publisher("/uav0/camera/set_hfov", Float64, queue_size=100)
		self.uav1_hfov_pub = rospy.Publisher("/uav1/camera/set_hfov", Float64, queue_size=100)
		self.uav2_hfov_pub = rospy.Publisher("/uav2/camera/set_hfov", Float64, queue_size=100)
		self.targetTruth_pub = rospy.Publisher("/TargetGroundTruth", Pose, queue_size=100)
		self.targetEstimation_pub = rospy.Publisher("/TargetEstimation", Pose, queue_size=100)

		self.states_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.States_callback, queue_size = 100)
		self.uav0_camerainfo_sub = rospy.Subscriber("/uav0/camera/rgb/camera_info", CameraInfo, self.CameraInfo0_callback, queue_size = 100)
		self.uav1_camerainfo_sub = rospy.Subscriber("/uav1/camera/rgb/camera_info", CameraInfo, self.CameraInfo1_callback, queue_size = 100)
		self.uav2_camerainfo_sub = rospy.Subscriber("/uav2/camera/rgb/camera_info", CameraInfo, self.CameraInfo2_callback, queue_size = 100)
		self.uav0_recognition_sub = rospy.Subscriber("/uav0/darknet_ros/bounding_boxes", BoundingBoxes, self.Recognition0_callback, queue_size = 100)
		self.uav0_recognition_sub = rospy.Subscriber("/uav1/darknet_ros/bounding_boxes", BoundingBoxes, self.Recognition1_callback, queue_size = 100)
		self.uav0_recognition_sub = rospy.Subscriber("/uav2/darknet_ros/bounding_boxes", BoundingBoxes, self.Recognition2_callback, queue_size = 100)

		# Parameter
		self.uav = uav
		self.pixel_size = 2.9*1e-6
		self.image_width = 1920
		self.FL_1x = 1921.176 # 6.5mm/2.9um
		self.FL_Curr = 1921.176

		# Variable
		self.img0, self.img1 ,self.img2 = None, None, None
		self.camerainfo0, self.camerainfo1, self.camerainfo2 = None, None, None
		self.P0, self.P0o, self.P1, self.P1o, self.P2, self.P2o, self.P3 = None, None, None, None, None, None, None

	def States_callback(self, msg):

		UAV_index = [msg.name.index('uav0'), msg.name.index('uav1'), msg.name.index('uav2'), msg.name.index('uav3')]

		P0 = np.array([msg.pose[UAV_index[0]].position.x, msg.pose[UAV_index[0]].position.y, msg.pose[UAV_index[0]].position.z])
		P0o = np.array([msg.pose[UAV_index[0]].orientation.x, msg.pose[UAV_index[0]].orientation.y,\
						msg.pose[UAV_index[0]].orientation.z, msg.pose[UAV_index[0]].orientation.w])

		P1 = np.array([msg.pose[UAV_index[1]].position.x, msg.pose[UAV_index[1]].position.y, msg.pose[UAV_index[1]].position.z])
		P1o = np.array([msg.pose[UAV_index[1]].orientation.x, msg.pose[UAV_index[1]].orientation.y,\
						msg.pose[UAV_index[1]].orientation.z, msg.pose[UAV_index[1]].orientation.w])

		P2 = np.array([msg.pose[UAV_index[2]].position.x, msg.pose[UAV_index[2]].position.y, msg.pose[UAV_index[2]].position.z])
		P2o = np.array([msg.pose[UAV_index[2]].orientation.x, msg.pose[UAV_index[2]].orientation.y,\
						msg.pose[UAV_index[2]].orientation.z, msg.pose[UAV_index[2]].orientation.w])

		P3 = np.array([msg.pose[UAV_index[3]].position.x, msg.pose[UAV_index[3]].position.y, msg.pose[UAV_index[3]].position.z])

		self.P0, self.P0rpw, self.P1, self.P1rpw, self.P2, self.P2rpw, self.P3 = \
			P0, self.q2rpy(P0o), P1, self.q2rpy(P1o), P2, self.q2rpy(P2o), P3
		self.P = np.array([self.P0, self.P1, self.P2])

	def q2rpy(self, q):

		if isinstance(q, Quaternion):

			rotate_x_rad = q.yaw_pitch_roll[2]
			rotate_y_rad = q.yaw_pitch_roll[1]
			rotate_z_rad = q.yaw_pitch_roll[0]
			R = q.rotation_matrix 
		else:

			q_ = Quaternion(q[3], q[0], q[1], q[2])
			rotate_x_rad = q_.yaw_pitch_roll[2]
			rotate_y_rad = q_.yaw_pitch_roll[1]
			rotate_z_rad = q_.yaw_pitch_roll[0]
			R = q_.rotation_matrix 

		# return np.array([rotate_x_rad, rotate_y_rad, rotate_z_rad])
		return R

	def CameraInfo0_callback(self, msg):

		fx = msg.K[0]
		fy = msg.K[4]
		cx = msg.K[2]
		cy = msg.K[5]

		self.camerainfo0 = np.array([fx, fy, cx, cy])

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

	def Recognition0_callback(self, msg):

		xmin = msg.bounding_boxes[0].xmin
		xmax = msg.bounding_boxes[0].xmax
		ymin = msg.bounding_boxes[0].ymin
		ymax = msg.bounding_boxes[0].ymax

		# Pixel frame
		u = (xmin+xmax)/2
		v = (ymin+ymax)/2
		# print("UAV0 - u&v: ", u, v)

		# Pixel frame convert to image plane (in meter)
		# self.img0 = np.array([(u-self.camerainfo0[2])*self.pixel_size, 
		# 						(v-self.camerainfo0[3])*self.pixel_size, 
		# 						(self.FL_Curr)*self.pixel_size])
		self.img0 = np.array([(u-self.camerainfo0[2]), (v-self.camerainfo0[3]), (self.FL_Curr)])
		# print("img0: ", self.img0)
		self.img0 /= np.linalg.norm(self.img0)
		# print("img0 unit: ", self.img0)

	def Recognition1_callback(self, msg):

		xmin = msg.bounding_boxes[0].xmin
		xmax = msg.bounding_boxes[0].xmax
		ymin = msg.bounding_boxes[0].ymin
		ymax = msg.bounding_boxes[0].ymax

		# Pixel frame
		u = (xmin+xmax)/2
		v = (ymin+ymax)/2

		# Pixel frame convert to image plane (in meter)
		# self.img1 = np.array([(u-self.camerainfo1[2])*self.pixel_size, 
		# 						(v-self.camerainfo1[3])*self.pixel_size, 
		# 						(self.FL_Curr)*self.pixel_size])
		self.img1 = np.array([(u-self.camerainfo1[2]), (v-self.camerainfo1[3]), (self.FL_Curr)])
		self.img1 /= np.linalg.norm(self.img1)

	def Recognition2_callback(self, msg):

		xmin = msg.bounding_boxes[0].xmin
		xmax = msg.bounding_boxes[0].xmax
		ymin = msg.bounding_boxes[0].ymin
		ymax = msg.bounding_boxes[0].ymax

		# Pixel frame
		u = (xmin+xmax)/2
		v = (ymin+ymax)/2

		# Pixel frame convert to image plane (in meter)
		self.img2 = np.array([(u-self.camerainfo2[2])*self.pixel_size, 
								(v-self.camerainfo2[3])*self.pixel_size, 
								(self.FL_Curr)*self.pixel_size])
		self.img2 = np.array([(u-self.camerainfo2[2]), (v-self.camerainfo2[3]), (self.FL_Curr)])
		self.img2 /= np.linalg.norm(self.img2)

	def SetCameraZoom(self, zoom):

		fx = zoom*self.FL_1x
		fy = fx
		hfov = 2*np.arctan2(self.image_width, 2*fx)

		output = Float64(data = hfov)

		self.uav0_hfov_pub.publish(output)
		self.uav1_hfov_pub.publish(output)
		self.uav2_hfov_pub.publish(output)

		self.FL_Curr = zoom*self.FL_1x

	def Main(self):

		Rx = np.array([[1, 0, 0],
					[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
					[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
		Rz = np.array([[np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi), 0],
					[np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi), 0],
					[0, 0, 1]])

		# Rotate from image plane(camera frame) to body frame
		tau0 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img0, (3,1))))
		tau1 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img1, (3,1))))
		tau2 = np.matmul(Rz, np.matmul(Rx, np.reshape(self.img2, (3,1))))
		tau = [tau0, tau1, tau2]
		# print("tau: ", tau)
		# print("tau0: ", tau0)

		# Rotation matrices from body frame to interial
		R0 = self.P0rpw #np.linalg.inv(self.P0rpw)
		R1 = self.P1rpw #np.linalg.inv(self.P1rpw)
		R2 = self.P2rpw #np.linalg.inv(self.P2rpw)
		R = [R0, R1, R2]
		# print("R: ", R)
		# print("result: ", np.allclose(np.matmul(R0, self.P0rpw), np.eye(3)))

		# Optimization - A
		A = np.array([])
		for i in range(3):

			hold = np.array([])

			for j in range(3):

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
		for i in range(3):

			if i == 0:

				B = -np.reshape(self.P[i], (3,1))
			else:

				B = np.vstack((B, -np.reshape(self.P[i], (3,1))))
		# print("B: ", B)
		# print("B shape: ", np.shape(B))

		# Solution - X
		# print("A: ", A)
		# print("A.T: ", A.T)
		X = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B))
		# print("X: ", X)
		Pt = -np.reshape(X[3:6], (1,3))[0]
		print("Pt: ", Pt)
		print("P3: ", self.P3)
		print("Error: ", np.linalg.norm(Pt-self.P3))

		output_ = Pose()
		output_.position.x = self.P3[0]
		output_.position.y = self.P3[1]
		output_.position.z = self.P3[2]
		self.targetTruth_pub.publish(output_)

		output_ = Pose()
		output_.position.x = Pt[0]
		output_.position.y = Pt[1]
		output_.position.z = Pt[2]
		self.targetEstimation_pub.publish(output_)


if __name__ == '__main__':

	rospy.init_node('Estimator', anonymous=True)

	node = DepthEstimator(3)

	rate = rospy.Rate(100)

	while (node.P0 is None) or (node.P1 is None) or (node.P2 is None) or (node.P3 is None) or\
			(node.camerainfo0 is None) or (node.camerainfo1 is None) or (node.camerainfo2 is None) or\
			(node.img0 is None) or (node.img1 is None) or (node.img2 is None):

			rate = rospy.Rate(100)

	rospy.loginfo("Estimator StandBy")

	while not rospy.is_shutdown():

		node.SetCameraZoom(2)
		node.Main()

		rate.sleep()