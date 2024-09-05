#!/usr/bin/python3

import rospy
import numpy as np
# import gurobipy as gp
# from gurobipy import GRB
from time import sleep, time
from cvxopt import matrix, solvers
from px4_mavros import Px4Controller
from gazebo_msgs.msg import ModelStates
from math import sin,cos,sqrt,atan2,acos,pi
from geometry_msgs.msg import Pose, Twist, TwistStamped

solvers.options['show_progress'] = False

PO, P1, P2, P3, P4, P5, P6, P7, P8, P9, A, b = None,None,None,None,None,None,None,None,None,None,None,None
P5v, P6v, P7v, P8v, P9v = None,None,None,None,None
cmd_vel = Twist()
d_safe = 1.0
m,x = None,None
currP8 = np.zeros(3)

def odom(msg):
	global PO, P1, P2, P3, P4, P5, P6, P7, P8, P9, A, b
	global P5v, P6v, P7v, P8v, P9v
	
	UAV0_index = msg.name.index('uav0')
	UAV1_index = msg.name.index('uav1')
	UAV2_index = msg.name.index('uav2')
	UAV3_index = msg.name.index('uav3')
	UAV4_index = msg.name.index('uav4')
	UAV5_index = msg.name.index('uav5')
	UAV6_index = msg.name.index('uav6')
	UAV7_index = msg.name.index('uav7')
	UAV8_index = msg.name.index('uav8')
	UAV9_index = msg.name.index('uav9')

	P0 = np.array([msg.pose[UAV0_index].position.x, msg.pose[UAV0_index].position.y, msg.pose[UAV0_index].position.z])
	P1 = np.array([msg.pose[UAV1_index].position.x, msg.pose[UAV1_index].position.y, msg.pose[UAV1_index].position.z])
	P2 = np.array([msg.pose[UAV2_index].position.x, msg.pose[UAV2_index].position.y, msg.pose[UAV2_index].position.z])
	P3 = np.array([msg.pose[UAV3_index].position.x, msg.pose[UAV3_index].position.y, msg.pose[UAV3_index].position.z])
	P4 = np.array([msg.pose[UAV4_index].position.x, msg.pose[UAV4_index].position.y, msg.pose[UAV4_index].position.z])

	P5 = np.array([msg.pose[UAV5_index].position.x, msg.pose[UAV5_index].position.y, msg.pose[UAV5_index].position.z])
	P5v = np.array([msg.twist[UAV5_index].linear.x, msg.twist[UAV5_index].linear.y, msg.twist[UAV5_index].linear.z])
	P6 = np.array([msg.pose[UAV6_index].position.x, msg.pose[UAV6_index].position.y, msg.pose[UAV6_index].position.z])
	P6v = np.array([msg.twist[UAV6_index].linear.x, msg.twist[UAV6_index].linear.y, msg.twist[UAV6_index].linear.z])
	P7 = np.array([msg.pose[UAV7_index].position.x, msg.pose[UAV7_index].position.y, msg.pose[UAV7_index].position.z])
	P7v = np.array([msg.twist[UAV7_index].linear.x, msg.twist[UAV7_index].linear.y, msg.twist[UAV7_index].linear.z])
	P8 = np.array([msg.pose[UAV8_index].position.x, msg.pose[UAV8_index].position.y, msg.pose[UAV8_index].position.z])
	P8v = np.array([msg.twist[UAV8_index].linear.x, msg.twist[UAV8_index].linear.y, msg.twist[UAV8_index].linear.z])
	P9 = np.array([msg.pose[UAV9_index].position.x, msg.pose[UAV9_index].position.y, msg.pose[UAV9_index].position.z])
	P9v = np.array([msg.twist[UAV9_index].linear.x, msg.twist[UAV9_index].linear.y, msg.twist[UAV9_index].linear.z])

	# PO = np.array([msg.pose[obs_index].position.x, msg.pose[obs_index].position.y, msg.pose[obs_index].position.z])
	# PO = np.array([100, 100, 100])

	# A = np.array([ \
	# 			  (-2*(P4-PO)[:2]).tolist() \
	# 			  ])

	# b = np.array([ \
	# 			  np.linalg.norm((P4-PO)[:2])**2 - d_safe**2 \
	# 			  ])

# def qp_ini():
# 	global m,x
	
# 	m = gp.Model("qp")
# 	m.setParam("NonConvex", 2.0)
# 	m.setParam("LogToConsole",0)
# 	x = m.addVars(2,ub=0.5, lb=-0.5, name="x")

# def addCons(i):
# 	global m

# 	m.addConstr(A[i,0]*x[0] + A[i,1]*x[1] <= b[i], "c"+str(i))

def vel(msg):

	global currP8

	if abs(msg.twist.linear.x) > 1e-6 and  abs(msg.twist.linear.y) > 1e-6:

		currP8 = P8

def	controller(time_):

	global cmd_vel, currP8, P8
	move_gain = 5
	delay_time = 1e-4
	# tra = [3*cos(t*pi), 3*sin(t*pi)]

	# if time_ > rospy.Duration(40.00) and time_ <= rospy.Duration(130.00):
	# 	tra = [P4[0] + 0.00*move_gain, P4[1] + 0.01*move_gain]
	# 	rospy.sleep(rospy.Duration(delay_time))
	# elif time_ > rospy.Duration(70.00) and time_ <= rospy.Duration(110.00):

	# 	tra = [P4[0] + 0.05*move_gain, P4[1] - 0.008*move_gain]
	# 	rospy.sleep(rospy.Duration(delay_time))
	# elif time_ > rospy.Duration(110.00) and time_ <= rospy.Duration(150.00):

	# 	tra = [P4[0] + 0.007*move_gain, P4[1] - 0.058*move_gain]
	# 	rospy.sleep(rospy.Duration(delay_time))
	# else:
	# 	tra = [P4[0], P4[1]]

	# u_des = np.array([1*((tra[0] - P4[0]) + 0),\
	# 		1*((tra[1] - P4[1]) + 0),\
	# 		20 - P4[2]])

	u_des = np.array([1*((currP8[0] - P8[0])), 1*((currP8[1] - P8[1])), 9.8 - P8[2]])

	# obj = (x[0] - u_des[0])**2 + (x[1] - u_des[1])**2
	# m.setObjective(obj)

	# m.remove(m.getConstrs())

	# for i in range (b.size):
	# 	addCons(i)

	# m.optimize()
	# u_opt = m.getVars()

	cmd_vel.linear.x = 0.5*u_des[0]
	cmd_vel.linear.y = 0.5*u_des[1]
	cmd_vel.linear.z = u_des[2]

	px4.vel_control(cmd_vel)

def Path_Follower(t):

	global P5, P6, P7, P8, P9, P5v, P6v, P7v, P8v, P9v
	u_mag = 1.0

	u_des = np.array([0.0,0.0,0.0])

	if t < 20:

		u_des = 2.0*np.array([(P9[0]-P8[0])-1.3+P9v[0] + P7[0]-P8[0], (P9[1]-P8[1])+1.0+P9v[1] + P7[1]-P8[1]+2.0, 2*(9.8 - P8[2])])
		u_mag = 0.5
	elif t < 45:
		u_des = np.array([0.3*(0.1), -10*(0.1), 9.8 - P8[2]])

		u_mag = 0.4
	else:

		u_des = np.array([0.5*((P5[0]-P8[0]) - 1.1 + P5v[0]),\
						0.5*((P5[1]-P8[1]) + 0.0 + P5v[1]),\
						2*(9.8 - P8[2])])

		u_mag = 0.5

	# CBF
	# neighbors_pos = np.array([P5, P6, P7, P9])
	# neighbors_vel = np.array([P5v, P6v, P7v, P9v])
	# dt_min = 0.35
	# communication_range = 1.0
	# G, H = [], []

	# for (p, v) in zip(neighbors_pos, neighbors_vel):

	# 	dist = np.linalg.norm(P8 - p)

	# 	if dist < communication_range:

	# 		d = (P8 - p)
	# 		h = 1.0*(dist**2 - dt_min**2) - 2*np.dot(d, v)
	# 		G.append([-2*d[0], -2*d[1]])
	# 		H.append(h)

	# if len(G) > 0 and len(H) > 0:

	# 	if len(G) == 1:

	# 		# QP
	# 		Q = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
	# 		p = matrix([-2*u_des[0], -2*u_des[1]])
	# 		G = matrix(G, (1,2))
	# 		H = matrix(H)
	# 		# print("G: ", G)
	# 		# print("H: ", H)
	# 	else:

	# 		# QP
	# 		Q = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
	# 		p = matrix([-2*u_des[0], -2*u_des[1]])
	# 		G = matrix(np.transpose(G))
	# 		G = G.trans()
	# 		H = matrix(H)
	# 		# print("G: ", G)
	# 		# print("H: ", H)

	# 	sol = solvers.coneqp(Q, p, G, H)
	# 	control_input = np.array([sol["x"][0], sol["x"][1]])
	# 	u_des = np.array([control_input[0], control_input[1], 2*(9.8 - P6[2])])

	# u_mag = np.linalg.norm(u_des)
	u_dir = u_des/np.linalg.norm(u_des)

	print("P8: ", P8)
	print("P8 u_mag: ", u_mag)
	print("P8 u_dir: ", u_dir)

	cmd_vel.linear.x = 1.0*u_mag*u_dir[0]
	cmd_vel.linear.y = 1.0*u_mag*u_dir[1]
	cmd_vel.linear.z = u_des[2]
	px4.vel_control(cmd_vel)

if __name__ == '__main__':

	try:
		rospy.init_node('controller_8')
		px4 = Px4Controller("uav8")
		rospy.Subscriber('/gazebo/model_states', ModelStates, odom, queue_size=10)
		rospy.Subscriber('/uav8/cmd_vel', TwistStamped, vel, queue_size=10)
		rate = rospy.Rate(100)

		while P8 is None:

			rate.sleep()

		currP8 = P8

		# qp_ini()
		last = time()

		while not rospy.is_shutdown():

			controller(np.round(time() - last, 2))
			
			print("Time Now: ", np.round(time() - last, 2))
			# Path_Follower(np.round(time() - last, 2))

			rate.sleep()
	except rospy.ROSInterruptException:

		pass
