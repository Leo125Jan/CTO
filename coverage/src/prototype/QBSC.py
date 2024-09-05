#!/usr/bin/python3

import sys
import csv
import rospy
import random
import numpy as np
import numexpr as ne
from time import sleep, time
from nav_msgs.msg import Odometry
from matplotlib.path import Path
from scipy.integrate import quad
from scipy import ndimage, sparse
from cvxopt import matrix, solvers
from shapely.geometry import Point
from collections import namedtuple
from scipy.optimize import linprog
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points
from math import sin, cos, tan, sqrt, atan2, acos, pi, exp
from std_msgs.msg import Int64, Float64MultiArray, MultiArrayLayout, MultiArrayDimension


solvers.options['show_progress'] = False

class PTZcon():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 50, Ka = 3, Kp = 3, step = 0.3):

		# Publisher & Subscriber | properties['ecode'] + 
		self.Curr_vel_pub = rospy.Publisher(properties['ecode'] + "/CurrVel", Float64MultiArray, queue_size = 100)
		self.Target_Resdiual_pub  = rospy.Publisher(properties['ecode'] + "/targetResdiual", Float64MultiArray, queue_size = 100)
		self.Sweet_Spot_pub  = rospy.Publisher(properties['ecode'] + "/SweetSpot", Float64MultiArray, queue_size = 100)
		self.Winnig_Bid_pub  = rospy.Publisher(properties['ecode'] + "/WinBid", Float64MultiArray, queue_size = 100)
		self.local_taskcomplete_pub  = rospy.Publisher(properties['ecode'] + "/TaskComplete", Float64MultiArray, queue_size = 100)
		self.local_resetready_pub  = rospy.Publisher(properties['ecode'] + "/ResetReady", Float64MultiArray, queue_size = 100)
		self.local_resetcomplete_pub  = rospy.Publisher(properties['ecode'] + "/ResetComplete", Float64MultiArray, queue_size = 100)
		self.counting_pub = rospy.Publisher(properties['ecode'] + "/Counting", Int64, queue_size = 100)
		self.target_pub  = rospy.Publisher(properties['ecode'] + "/Ttarget", Odometry, queue_size = 100)

		# Variable of PTZ
		self.grid_size = grid_size
		self.map_size = map_size
		self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))
		self.id = properties['id']
		self.pos = properties['position']
		self.perspective = properties['perspective']/np.linalg.norm(properties['perspective'])
		self.alpha = properties['AngleofView']/180*np.pi
		self.R = properties['range_limit']
		self.lamb = properties['lambda']
		self.color = properties['color']
		self.r = 0
		self.top = 0
		self.ltop = 0
		self.rtop = 0
		self.H = 0
		self.centroid = np.array([0,0])
		self.R_max = (self.lamb + 1)/(self.lamb)*self.R*np.cos(self.alpha)
		self.sweet_spot = np.array([None, None])

		x_range = np.arange(0, self.map_size[0], self.grid_size[0])
		y_range = np.arange(0, self.map_size[1], self.grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		self.W = W.transpose()

		# Inscribed circle of FOV
		self.head_theta = np.arctan(abs(self.perspective[1]/self.perspective[0]))
		self.incircle_r = (self.R_max*np.sin(self.alpha))/(1 + np.sin(self.alpha))
		self.incircle_x = self.pos[0] + (self.R_max - self.incircle_r)*np.sign(self.perspective[0])*np.cos(self.head_theta)
		self.incircle_y = self.pos[1] + (self.R_max - self.incircle_r)*np.sign(self.perspective[1])*np.sin(self.head_theta)
		self.incircle_A = np.pi*self.incircle_r**2

		self.incircle = [(self.incircle_x, self.incircle_y), self.incircle_r, self.incircle_A]
		
		# Decentralized Communicaton
		self.neighbor_notfication = None
		self.comc = None
		self.c, self.x, self.y, self.h, self.J = None, None, None, None, None
		self.last_cluster_set, self.last_teammate = [None], [None]
		self.Jx = None
		self.self_reset, self.consensus_reset = True, False
		self.task_complete, self.reset_complete, self.reset_ready, self.reset_ready = False, False, False, False
		self.flag = 0

		# self.max_speed = properties['max_speed']
		self.translation_force = 0  # dynamics of positional changes
		self.perspective_force = 0  # dynamics of changing perspective direction
		self.stage = 1              # 1: Free player 2: Occupied Player 3: Cooperative player
		self.target = None
		self.virtual_target = self.R*cos(self.alpha)*self.perspective
		self.target_assigned = -1
		self.step = step

		self.FoV = np.zeros(self.size)
		self.Kv = Kv                # control gain for perspective control law toward voronoi cell
		self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
		self.Kp = Kp                # control gain for positional change toward voronoi cell 
		self.event = np.zeros((self.size[0], self.size[1]))

	def UpdateState(self, targets, neighbors, states, time_):

		self.time = time_
		self.targets_pos = targets["2DPosition"]
		self.targets_3dpos = targets["3DPosition"]
		self.targets_vel = targets["Vel"]
		self.neighbors = neighbors
		self.pos = states["Position"]
		self.perspective = states["Perspective"]
		self.one_hop_neighbor = states["NeighorIndex"]
		self.Pd = states["Pd"]
		self.cluster_set = states["ClusterSet"]
		self.col_ind = states["ColInd"]
		self.teammate = states["Teammate"]
		self.cluster_labels = None

		print("id: " + str(self.id))

		# Field of View
		self.UpdateFoV()
		self.polygon_FOV()
		self.UpdateLocalVoronoi()

		# CBAA
		self.CBAA(self.targets_pos, time_)

		# Destination Point Determination
		self.Gradient_Descent(self.targets_pos, self.Pd, self.cluster_set, self.col_ind, self.cluster_labels, time_)

		# Control Law
		self.Gradient_Ascent(self.targets_pos, time_)
		# self.FormationControl(self.targets_pos)
		self.CBF_Collision_Herding(self.targets_pos)
		self.UpdateOrientation()
		self.UpdateZoomLevel()
		self.UpdatePosition()
		self.UpdateSweetSpot()

		# Boardcast Data
		self.PublishMessage()

		print("id: " + str(self.id), "\n")

		return self.translational_force, self.perspective_force, self.step

	def CBAA(self, targets, time_):

		if len(self.teammate) == 1:

			self.Jx = None
			self.self_reset = True
			print("CBAA id: " + str(self.id) + ", no teammate")
		else:

			''' ---------- Receive Message ---------- '''
			# Neighbor Reset complete List
			Rc_list = np.array([self.neighbors["ResetComplete"][str(key)] for key in self.one_hop_neighbor])
			Rc_list = np.append(Rc_list, self.reset_complete)
			# print("CBAA id, Rc_list: ", self.id, Rc_list)

			# Neighbor Task complete List
			Tc_list = np.array([self.neighbors["TaskComplete"][str(key)] for key in self.one_hop_neighbor])
			Tc_list = np.append(Tc_list, self.task_complete)
			# print("CBAA id, Tc_list: ", self.id, Tc_list)

			# Neighbor Reset ready List
			Rr_list = np.array([self.neighbors["ResetReady"][str(key)] for key in self.one_hop_neighbor])
			Rr_list = np.append(Rr_list, self.reset_ready)
			# print("CBAA id, Rr_list: ", self.id, Rr_list)

			# Counting Mechanism List
			respond_node = np.array([self.neighbors["Counting"][str(key)] for key in self.one_hop_neighbor])
			# print("CBAA id, respond_node: ", self.id, respond_node)

			# Initialization
			if (sorted(self.last_cluster_set) != sorted(self.cluster_set)) or self.self_reset or\
				(sorted(self.last_teammate) != sorted(self.teammate)):
			# if (sorted(self.last_cluster_set) != sorted(self.cluster_set)):

				# print("Init")
				self.x = np.zeros(len(self.cluster_set))
				self.y = np.ones(len(self.cluster_set))*np.inf
				# print("CBAA id, init x: ", self.id, self.x)
				# print("CBAA id, init y: ", self.id, self.y)
				# print("CBAA id, cluster_set: ", self.id, self.cluster_set)

				targets_position = np.array([targets[i][0] for i in range(len(targets))])
				task = targets_position[self.cluster_set]

				self.c = np.linalg.norm(task-self.pos, axis=1)
				# print("init c: ", self.c)

				self.task_complete = False
				self.reset_complete = True
				self.reset_ready = False
				self.self_reset = False
				self.Jx = None

				self.reset_protect = True

			if ((self.reset_protect) and (np.all(Rc_list) == False)) or\
				(sorted(self.last_cluster_set) != sorted(self.cluster_set)) or\
				(sorted(self.last_teammate) != sorted(self.teammate)):

				n = len(self.teammate) - 1

				# Neighbor Reset complete List
				Rc_list = np.array([False for _ in range(n)])

				# Neighbor Task complete List
				Tc_list = np.array([False for _ in range(n)])

				# Neighbor Reset ready List
				Rr_list = np.array([False for _ in range(n)])

				# Counting Mechanism List
				respond_node = np.array([None for _ in range(n)])
			else:

				self.reset_protect = False


			''' ---------- CBAA ----------'''
			# Auction Process
			if np.sum(self.x) == 0:

				targets_position = np.array([targets[i][0] for i in range(len(targets))])
				task = targets_position[self.cluster_set]

				self.c = np.linalg.norm(task-self.pos, axis=1)

				self.h = np.where(self.c < self.y, 1, 1e3)
				self.J = np.argmin(self.h*self.c)
				# print("auction h: ", self.h)
				# print("auction J: ", self.J)
				self.x[self.J] = 1
				self.y[self.J] = self.c[self.J]
			
			# print("CBAA id, auction x: ", self.id, self.x)
			# print("CBAA id, auction y: ", self.id, self.y)

			# Neighbor's Wnning Bid
			for key in self.one_hop_neighbor:

				if (self.neighbors["WinBid"][str(key)] != None):

					if (len(self.neighbors["WinBid"][str(key)]) != len(self.y)):

						self.neighbors["WinBid"][str(key)] = None

			# Consensus Process
			count = 0
			for key in self.one_hop_neighbor:

				if self.neighbors["WinBid"][str(key)] == None:

					count += 1

			if count > 0:

				self.task_complete = False
			else:

				# for key in self.one_hop_neighbor:

				# 	print("key, Winbid: ", key, self.neighbors["WinBid"][str(key)])

				win_bid = np.array([self.neighbors["WinBid"][str(key)] for key in self.one_hop_neighbor])
				# print("CBAA id, win_bid: ", self.id, win_bid)

				temp = np.vstack((self.y.copy(), win_bid))
				# print("CBAA id, temp: ", self.id, temp)
				self.y = np.min(temp, axis=0)
				# print("CBAA id, consensus y: ", self.id, self.y)

				smallest_number = np.min(temp[:, self.J])
				# print("CBAA id, smallest_number: ", self.id, smallest_number)
				indices = np.where(temp[:, self.J] == smallest_number)[0]
				# print("CBAA id, indices: ", self.id, indices)

				if len(indices) == 1:

					z = np.argmin(temp[:, self.J])
					# print("z: ", z)

					if z != 0:

						self.x[self.J] = 0
						self.task_complete = False
					else:

						self.task_complete = True
				else:

					if self.y[self.J] == smallest_number:

						self.task_complete = True
					else:

						self.x[self.J] = 0
						self.task_complete = False

			self.last_cluster_set = self.cluster_set.copy()
			self.last_teammate = self.teammate.copy()

			if self.task_complete:

				self.Jx = self.x.copy()

			# print("CBAA id, consensus x: ", self.id, self.x)
			# print("CBAA id, consensus Jx: ", self.id, self.Jx)


			''' ---------- Consensus Mechanism ---------- '''
			# Neighbor Reset complete List
			# Rc_list = np.array([self.neighbors["ResetComplete"][str(key)] for key in self.one_hop_neighbor])
			# Rc_list = np.append(Rc_list, self.reset_complete)
			# print("CBAA id, Rc_list: ", self.id, Rc_list)

			# if np.all(Rc_list):

			# 	output_ = Float64MultiArray(data = [self.task_complete])
			# 	self.local_taskcomplete_pub.publish(output_)

			# Neighbor Task complete List
			# Tc_list = np.array([self.neighbors["TaskComplete"][str(key)] for key in self.one_hop_neighbor])
			# Tc_list = np.append(Tc_list, self.task_complete)
			# print("CBAA id, Tc_list: ", self.id, Tc_list)
			self.consensus_reset = np.all(Tc_list)
			# print("CBAA id, consensus_reset: ", self.id, self.consensus_reset)

			if self.consensus_reset:

				self.reset_ready = True
			else:

				self.reset_ready = False

			# output_ = Float64MultiArray(data = [self.reset_ready])
			# self.local_resetready_pub.publish(output_)

			# Neighbor Reset ready List
			# Rr_list = np.array([self.neighbors["ResetReady"][str(key)] for key in self.one_hop_neighbor])
			# Rr_list = np.append(Rr_list, self.reset_ready)
			# print("CBAA id, Rr_list: ", self.id, Rr_list)

			# Counting Mechanism
			if np.all(Rr_list):

				iteration = len(self.teammate)
				# print("iteration: ", iteration)
				# respond_node = np.array([self.neighbors["Counting"][str(key)] for key in self.one_hop_neighbor])

				if np.any(respond_node == None):

					self.flag = int(0)
					# output_ = Int64(data = self.flag)
					# self.counting_pub.publish(output_)
				else:

					if np.array_equal(respond_node, np.full(respond_node.shape, respond_node[0])):

						unique_number = np.unique(respond_node)[0]

						if unique_number == iteration:

							self.flag = unique_number
							output_ = Int64(data = self.flag)
							self.counting_pub.publish(output_)

							self.self_reset = True
							self.flag = 0
						else:

							if unique_number == iteration - 1:

								self.reset_complete = False

							self.flag = unique_number + 1
					else:
						
						unique_number = np.max(respond_node)

						self.flag = unique_number
				
				# output_ = Int64(data = self.flag)
				# self.counting_pub.publish(output_)
				# print("CBAA id, flag: ", self.id, self.flag)

			''' ---------- Receive Message ---------- '''
			# Reset Complete Publish
			output_ = Float64MultiArray(data = [self.reset_complete])
			self.local_resetcomplete_pub.publish(output_)

			# Winning Bid Publish
			output_ = Float64MultiArray(data = self.y)
			self.Winnig_Bid_pub.publish(output_)

			# Task Complete Publish
			if np.all(Rc_list):

				output_ = Float64MultiArray(data = [self.task_complete])
				self.local_taskcomplete_pub.publish(output_)

			# Reset Ready Publish
			output_ = Float64MultiArray(data = [self.reset_ready])
			self.local_resetready_pub.publish(output_)

			# Counting Mechanism Publish
			output_ = Int64(data = self.flag)
			self.counting_pub.publish(output_)

	def Hamiltonian_Path(self, targets, start_vertex):

		# print("start_vertex: ", start_vertex)

		points = [targets[i] for i in range(len(targets))]
		# print("points: " + str(points) + "\n")

		# Calculate the pairwise distances between targets
		distances = distance.cdist(points, points)
		# print("distances: " + str(distances) + "\n")

		num_vertices = np.shape(distances[0])[0]

		visited = [False]*np.ones(num_vertices)
		visited[start_vertex] = True
		temp_root = start_vertex

		mst_edges, mst_weights = [], []

		while len(mst_edges) < num_vertices - 1:

			min_edge = None
			min_weight = float('inf')

			for i in range(num_vertices):
				
				# if visited[i]:
				if i == temp_root:
					
					for j in range(num_vertices):

						if not visited[j] and distances[i, j] < min_weight:
							
							min_edge = (i, j)
							min_weight = distances[i, j]
			if min_edge:

				mst_edges.append(min_edge)
				mst_weights.append(min_weight)
				visited[min_edge[1]] = True
				temp_root = min_edge[1]

		# print("Hamiltonian Path: " + str(mst_edges) + "\n")
		# print("Weights: " + str(mst_weights) + "\n")

		return [mst_edges]

		# Define the weight threshold for deleting edges
		weight_threshold = 2.0*self.incircle[1]

		# print("weight_thershold: " + str(weight_threshold))

		modified_edges, modified_weights = [], []

		for edge, weight in zip(mst_edges, mst_weights):

			# Check if the weight of the edge exceeds the threshold
			if weight <= weight_threshold:

				# Add the edge to the modified minimum spanning tree
				modified_edges.append(edge)
				modified_weights.append(weight)

		modified_edges = [tuple(element) for element in modified_edges]

		# print("Modified Hamiltonian Path: " + str(modified_edges) + "\n")
		# print("Modified Weights: " + str(modified_weights) + "\n")

		# print(halt)
		# return [modified_edges]

		if len(modified_edges) == 0:

			Hamiltonian_Path = [[]]
		else:
			threshold = np.sqrt(2)
			continuous_groups = []
			current_group = []

			for i in range(len(modified_edges) - 1):

				# distance_ = np.linalg.norm(modified_edges[i] - modified_edges[i + 1])

				if modified_edges[i][1] == modified_edges[i+1][0]:

					current_group.append(modified_edges[i])
				else:

					current_group.append(modified_edges[i])
					continuous_groups.append(current_group)
					current_group = []

			# # Append the last point to the current group
			current_group.append(modified_edges[-1])
			continuous_groups.append(current_group)

			Hamiltonian_Path = [modified_edges]
			# Hamiltonian_Path = continuous_groups

		# print("Hamiltonian_Path: ", Hamiltonian_Path)

		return Hamiltonian_Path

	def Graph_Construction(self, I_vertex):

		# print(str(self.id) + " I_vertex: ", I_vertex)
		I_vertex = np.array(I_vertex, dtype=np.float64)

		if np.shape(I_vertex)[0] > 1:

			start_vertex = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
			# start_vertex = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.sweet_spot, axis=2))
			# print(str(self.id) + " self.sweet_spot: ", self.sweet_spot)
			# print("dist: ", np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
			# print("self.pos: ", self.pos)
			# print(str(self.id) + " start_vertex: " + str(start_vertex))
			# print(str(self.id) + " start: " + str(start))

			# if all(element == 0 for element in self.x):
			print("id, Jx: ", self.id, self.Jx)
			if np.all(self.Jx == None):

				start = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
			else:

				start = np.where(self.Jx == 1)[0][0]
			# print("start: ", start)

			edges = self.Hamiltonian_Path(I_vertex, start)
			# print("edges: " + str(edges))
			# print(np.shape(edges))

			if int(np.shape(edges)[0]) == 1:

				trunk_hold = edges[0]
				path = I_vertex

			if int(np.shape(edges)[0]) == 1 and int(np.shape(edges)[1]) == 0:

				# print("In")
				trunk_hold = [[-1]]
				path = [I_vertex[start_vertex]]

			if int(np.shape(edges)[0]) > 1:

				max_element = max(edges, key=len)
				# print("Element with maximum size:", max_element)

				# Find elements with identical sizes
				size_count = {}
				size_count[len(max_element)] = [max_element]
				# print("size_count: " + str(size_count))

				for element in edges:

					size = len(element)

					if size == len(max_element) and not np.all(np.array(element) == np.array(size_count[len(max_element)])):

						size_count[size].append(element)

				# print("size_count: " + str(size_count))

				# Filter elements with identical sizes
				identical_size_elements = [elements for size, elements in size_count.items()][0]
				# print("Elements with identical sizes:", identical_size_elements)

				if np.shape(identical_size_elements)[0] == 1:

					trunk_hold = identical_size_elements[0]
					path = I_vertex
				else:

					dist = []

					for i in range(np.shape(identical_size_elements)[0]):

						target_points = []

						for element in identical_size_elements:

							start = I_vertex[branch[0]]
							end = I_vertex[branch[1]]

							if any(np.array_equal(start, arr) for arr in target_points) == False and len(target_points) > 0:

								target_points.append(start)
							elif len(target_points) == 0:

								target_points.append(start)

							if any(np.array_equal(end, arr) for arr in target_points) == False and len(target_points) > 0:

								target_points.append(end)
							elif len(target_points) == 0:

								target_points.append(end)

						nodes = target_points[0:np.shape(target_points)[0]-i]
						x = [element[0] for element in nodes]; avg_x = np.mean(x)
						y = [element[1] for element in nodes]; avg_y = np.mean(y)
						
						geometric_center = np.array([(avg_x, avg_y)])
						distance_ = np.linalg.norm(geometric_center - self.sweet_spot)
						dist.append(distance_)

					trunk_hold = identical_size_elements[np.argmin(dist)]
					path = I_vertex
		else:

			trunk_hold = [[-1]]
			path = I_vertex

		# print("path: ", path)
		# print("trunk_hold: ", trunk_hold)

		return trunk_hold, path

	def Outsourcing(self, trunk_hold, path):

		if trunk_hold[0][0] != -1:

			nodes = []

			for branch in trunk_hold:

				start = path[branch[0]]
				end = path[branch[1]]
				
				if any(np.array_equal(start, arr) for arr in nodes) == False and len(nodes) > 0:

					nodes.append(start)
				elif len(nodes) == 0:

					nodes.append(start)

				if any(np.array_equal(end, arr) for arr in nodes) == False and len(nodes) > 0:

					nodes.append(end)
				elif len(nodes) == 0:

					nodes.append(end)

			# print("nodes: ", nodes)

			# Trunk after eliminate
			trunk_eli = []

			# Path after eliminate & Path Residual
			path_save = set([tuple((element)) for element in path])

			path_eli = []; path_eli.append(tuple(nodes[0]))
			path_resdiual = []

			for index_ in range(1, len(nodes)):

				# print("index_: ", index_)

				x = [nodes[i][0] for i in range(0, index_+1)]; avg_x = np.mean(x)
				y = [nodes[i][1] for i in range(0, index_+1)]; avg_y = np.mean(y)
				geometric_center = np.array([avg_x, avg_y])
				rangecircle_r, R_hold = 0, -np.inf

				for i in range(0, index_+1):

					p1 = geometric_center
					p2 = np.array([nodes[i][0], nodes[i][1]])

					distance_ = np.linalg.norm(p1 - p2)

					if distance_ >= R_hold:

						rangecircle_r = distance_
						R_hold = distance_

					# print("rangecircle_r: ", rangecircle_r)

				rangecircle_A = np.pi*(rangecircle_r)**2
				# print("rangecircle_A: ", rangecircle_A)
				# print("incircle_r: ", self.incircle[1])
				# print("incircle_A: ", self.incircle[2])

				if rangecircle_A >= 1.0*self.incircle[2]:

					break
				else:

					path_eli.append(tuple(nodes[index_]))
					trunk_eli.append(trunk_hold[index_-1])

			path_resdiual = np.array([element for element in path_save if element not in path_eli])
			# print("path_eli: ", path_eli)
			# print("path_resdiual: ", path_resdiual)

			# Communication Section
			# print("one_hop_neighbor: ", self.one_hop_neighbor)

			self.neighbor_notfication = False
			self.comc = {str(neighbor_id): [] for neighbor_id in self.one_hop_neighbor}

			# print("self.comc: ", self.comc)

			if len(path_resdiual) != 0:

				neighbor_pos = {}

				for (id_, i) in zip(self.neighbors["ID"], range(len(self.neighbors["ID"]))):

					if (int(id_) in self.one_hop_neighbor):

						# print("neighbors Swt: ", self.neighbors["Swt"])
						neighbor_pos[str(id_)] = self.neighbors["Swt"][str(id_)]

				# print("neighbor_pos: ", neighbor_pos)

				for element in path_resdiual:

					mini_ = None
					distance_hold = 8*self.incircle[1]

					for index_, member in neighbor_pos.items():

						dist_ = np.linalg.norm(member - element)

						if dist_ < distance_hold:

							mini_ = index_
							distance_hold = dist_

					if mini_ != None:

						self.comc[str(mini_)].append(element)

			# print("self.comc: ", self.comc)

			if self.comc != None:

				self.neighbor_notfication = True
			
			if self.neighbor_notfication:

				print("Neighbor Notification")
		else:
			
			self.comc = {str(neighbor_id): [] for neighbor_id in self.one_hop_neighbor}

	def Judgement_Algorithm(self, trunk_hold, path):

		# Judgement Algorithm
		if trunk_hold[0][0] == -1:

			trunk_hold_finale = [[-1]]
			path_finale = path
		else:

			nodes = []

			for branch in trunk_hold:

				start = path[branch[0]]
				end = path[branch[1]]
				
				if any(np.array_equal(start, arr) for arr in nodes) == False and len(nodes) > 0:

					nodes.append(start)
				elif len(nodes) == 0:

					nodes.append(start)

				if any(np.array_equal(end, arr) for arr in nodes) == False and len(nodes) > 0:

					nodes.append(end)
				elif len(nodes) == 0:

					nodes.append(end)

			# print("nodes: ", nodes)

			# Trunk after eliminate
			trunk_eli = []

			for index_ in range(1, len(nodes)):

				# print("index_: ", index_)

				x = [nodes[i][0] for i in range(0, index_+1)]; avg_x = np.mean(x)
				y = [nodes[i][1] for i in range(0, index_+1)]; avg_y = np.mean(y)
				geometric_center = np.array([avg_x, avg_y])
				rangecircle_r, R_hold = 0, -np.inf

				for i in range(0, index_+1):

					p1 = geometric_center
					p2 = np.array([nodes[i][0], nodes[i][1]])

					distance_ = np.linalg.norm(p1 - p2)

					if distance_ >= R_hold:

						rangecircle_r = distance_
						R_hold = distance_

					# print("rangecircle_r: ", rangecircle_r)

				rangecircle_A = np.pi*(rangecircle_r)**2
				# print("rangecircle_A: ", rangecircle_A)
				# print("incircle_r: ", self.incircle[1])
				# print("incircle_A: ", self.incircle[2])

				if rangecircle_A >= 1.0*self.incircle[2]:

					break
				else:

					trunk_eli.append(trunk_hold[index_-1])

			if len(trunk_eli) == 0:

				trunk_hold_finale = [[-1]]
				path_finale = path
			else:

				trunk_hold_finale = trunk_eli
				path_finale = path

		return trunk_hold_finale, path_finale

	def Optimal_Cluster(self, targets, Pd, cluster_set, col_ind, cluster_labels, time_):

		# Agglomerative Hierarchical Clustering
		targets_position = np.array([targets[i][0] for i in range(len(targets))])
		# print("target_position: ", targets_position)

		# watch_1 = col_ind[self.id]
		watch_1 = int(col_ind)
		# print("watch_1: ", watch_1)
		# print("cluster_set: ", cluster_set)

		if watch_1 >= 0:

			I_vertex = []
			# I_vertex = targets_position[cluster_set[watch_1]]
			I_vertex = targets_position[cluster_set]
			# self.Pd = Pd[watch_1]
		else:

			# true_watch_1 = int(watch_1) + 100
			true_watch_1 = cluster_set[0]
			I_vertex = np.array([np.array(targets_position[true_watch_1])])

		# print("I_vertex: ", I_vertex)

		# First Construction
		trunk_hold, path = self.Graph_Construction(I_vertex)
		# print("path_origin: ", path)
		# print("trunk_hold_origin: ", trunk_hold)
		
		# Call for Help
		self.Outsourcing(trunk_hold, path)

		# Combine neighbor message
		# print("Neighbors residual: ", self.neighbors["Residual"])
		check_list = np.array([[None, None]])
		# for residual in self.neighbors["Residual"]:
		for neighbor_id in self.one_hop_neighbor:

			# if neighbor.comc != None:
			# if (self.neighbors["Residual"][residual] != None).all() and self.neighbors["Residual"][residual] not in path:

				# path = np.concatenate((path, [self.neighbors["Residual"][str(neighbor_id)]]), axis = 0)

			if not np.array_equal(self.neighbors["Residual"][str(neighbor_id)], check_list):

				for point in self.neighbors["Residual"][str(neighbor_id)]:

					if not any(np.array_equal(point, row) for row in path):

						np.vstack((path,point))

		# print("path_combined: ", path)
		# print("trunk_hold_origin: ", trunk_hold)

		# Sceondly Construction
		trunk_hold, path = self.Graph_Construction(path)
		# print("path_evolution: ", path)
		# print("trunk_hold_evolution: ", trunk_hold)

		# Secondly Judgement
		trunk_hold, path = self.Judgement_Algorithm(trunk_hold, path)
		# print("path_finale: ", path)
		# print("trunk_hold_finale: ", trunk_hold)

		return path, trunk_hold

	def Gradient_Descent(self, targets, Pd, cluster_set, col_ind, cluster_labels, time_):

		# Configuration of calculation cost function
		self.sweet_spot = self.pos + self.R*np.cos(self.alpha)*self.perspective

		# Inscribed circle of FOV
		head_theta = np.arctan(abs(self.perspective[1]/self.perspective[0]))
		incircle_r = (self.R_max*np.sin(self.alpha))/(1 + np.sin(self.alpha))
		incircle_x = self.pos[0] + (self.R_max - incircle_r)*np.sign(self.perspective[0])*np.cos(head_theta)
		incircle_y = self.pos[1] + (self.R_max - incircle_r)*np.sign(self.perspective[1])*np.sin(head_theta)
		incircle_A = 0.8*1.0*np.pi*incircle_r**2

		self.incircle = [(incircle_x, incircle_y), incircle_r, incircle_A]

		# Optimal Clustering
		path, trunk = self.Optimal_Cluster(targets, Pd, cluster_set, col_ind, cluster_labels, time_)
		# print("path: " + str(path))
		# print("trunk: " + str(trunk))
		target_points = []

		if trunk[0][0] == -1:

			nearest_index = np.argmin(np.linalg.norm(path[:, np.newaxis] - self.pos, axis=2))

			dx = 1.0*\
				np.array([(path[nearest_index][0]-self.virtual_target[0]), (path[nearest_index][1]-self.virtual_target[1])])
			# self.target = [[path[0][0], 2.0, 10]]

			# print("target_1: " + str(self.target))

			# Herding Algorithm
			targets_position = np.array([target[0] for target in targets])
			GCM = np.mean(targets_position, axis = 0)
			cluster_center = np.mean(path, axis = 0)

			# print("GCM: ", GCM)
			# print("cluster_center: ", cluster_center)

			ra = 1
			df = (cluster_center - GCM)
			pc = GCM + df + (df/np.linalg.norm(df))*ra*np.sqrt(len(path))
			# pc = GCM + df + self.perspective*self.R*np.cos(self.alpha)
			# self.Pd = self.pos + [1e-3, 1e-3]
			# self.Pd = pc
			self.C = cluster_center
			self.GCM = GCM
		else:

			for branch in trunk:

				start = path[branch[0]]
				end = path[branch[1]]

				if any(np.array_equal(start, arr) for arr in target_points) == False and len(target_points) > 0:

					target_points.append(start)
				elif len(target_points) == 0:

					target_points.append(start)

				if any(np.array_equal(end, arr) for arr in target_points) == False and len(target_points) > 0:

					target_points.append(end)
				elif len(target_points) == 0:

					target_points.append(end)

			# Herding Algorithm
			targets_position = np.array([target[0] for target in targets])
			GCM = np.mean(targets_position, axis = 0)
			cluster_center = np.mean(target_points, axis = 0)

			# print("GCM: ", GCM)
			# print("cluster_center: ", cluster_center)

			ra = 1
			df = (cluster_center - GCM)
			pc = GCM + df + (df/np.linalg.norm(df))*ra*np.sqrt(len(path))
			# pc = GCM + df + self.perspective*self.R*np.cos(self.alpha)
			# self.Pd = self.pos + [1e-3, 1e-3]
			# self.Pd = pc
			self.C = cluster_center
			self.GCM = GCM

			# print("target_points: " + str(target_points))

			# if np.shape(target_points)[0] > 3:

			# 	nodes = target_points[0:np.shape(target_points)[0]]
			# 	x = [element[0] for element in nodes]; avg_x = np.mean(x)
			# 	y = [element[1] for element in nodes]; avg_y = np.mean(y)
				
			# 	geometric_center = np.array((avg_x, avg_y))
			# 	self.target = [[geometric_center, 2.0, 10]]
			# 	print("target_n: " + str(self.target))
			# elif np.shape(target_points)[0] == 3:

			# 	nodes = target_points[0:np.shape(target_points)[0]]

			# 	x = [element[0] for element in nodes]; avg_x = np.mean(x)
			# 	y = [element[1] for element in nodes]; avg_y = np.mean(y)
			# 	geometric_center = np.array((avg_x, avg_y))
			# 	self.target = [[geometric_center, 2.0, 10]]
			# 	print("target_3: " + str(self.target))
			# elif np.shape(target_points)[0] == 2:

			# 	nodes = target_points[0:np.shape(target_points)[0]]

			# 	p1 = np.array(nodes[0])
			# 	p2 = np.array(nodes[1])
			# 	sidecircle_center = np.array([0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])])
			# 	self.target = [[sidecircle_center, 2.0, 10]]
			# 	print("target_2: " + str(self.target))

			x, y, = 0, 0
			C_descent = []; Cd = 0; Cn_ = []; Dx = []
			dx = np.zeros(2)
			for element, i in zip(target_points, range(np.shape(target_points)[0])):

				if np.shape(target_points)[0] - i > 3:

					nodes = target_points[0:np.shape(target_points)[0]-i]
					x = [element[0] for element in nodes]; avg_x = np.mean(x)
					y = [element[1] for element in nodes]; avg_y = np.mean(y)
					
					geometric_center = np.array([(avg_x, avg_y)])
					rangecircle_r, R_hold = 0, -np.inf

					for (element, i) in zip(nodes, range(len(nodes))):

						p1 = geometric_center
						p2 = np.array([element[0], element[1]])

						distance_ = np.linalg.norm(p1 - p2)

						if distance_ >= R_hold:

							rangecircle_r = distance_
							R_hold = distance_

					rangecircle_A = np.pi*(rangecircle_r)**2
					theta = self.calculate_tangent_angle((avg_x, avg_y), rangecircle_r, self.pos)

					# print("rangecircle_A: " + str(rangecircle_A))
					# print("theta: " + str(theta))

					# Cn = np.exp( -( (rangecircle_A/(0.8*incircle_A))*(1/(2*0.5**2)) ) )*\
					# 	np.exp( -( ((theta)/(1*self.alpha))*(1/(2*0.5**2)) ) )
					# Cn = np.exp( -( (rangecircle_A/(1.0*incircle_A))*(1/(2*0.5**2)) ) )
					# Cn = 0.5*( 1 - np.tanh( 3.0*(rangecircle_A/(1.0*incircle_A)-0.5) ) )*\
					# 	0.5*( 1 - np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
					# Cn = 0.5*( 1 - np.tanh( 10.0*( (rangecircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.7 ) ) )
					Cn = 0.5*( 1 - np.tanh( 15.0*( (rangecircle_A/(1.0*incircle_A)) - 0.7 ) ) )

					# print("incircle_A: " + str(incircle_A))
					# print("Cn_4: " + str(Cn))

					if len(C_descent) == 0:

						# dx += (-Cn)*(-2)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])]))

						# Cd = np.exp( -( ((0.25*incircle_A)/rangecircle_A)*(1/(2*0.5**2)) ) )*\
						# 	np.exp( -( ((0.25*self.alpha)/(theta))*(1/(2*0.5**2)) ) )
						# Cd = np.exp( -( ((0.50*incircle_A)/rangecircle_A)*(1/(2*0.5**2)) ) )
						# Cd = 0.5*( 1 + np.tanh( 3.0*(rangecircle_A/(1.0*incircle_A)-0.5) ) )*\
						# 	0.5*( 1 + np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
						# Cd = 0.5*( 1 + np.tanh( 10.0*( (rangecircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.7) ) )
						Cd = 0.5*( 1 + np.tanh( 15.0*( (rangecircle_A/(1.0*incircle_A)) - 0.8) ) )
						C_descent.append(Cd)
					else:

						Cn *= C_descent[-1]
						# dx += (-Cn)*(-2)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(avg_x-self.virtual_target[0]), (avg_y-self.virtual_target[1])]))

						# Cd = np.exp( -( ((0.25*incircle_A)/rangecircle_A)*(1/(2*0.5**2)) ) )*\
						# 	np.exp( -( ((0.25*self.alpha)/(theta))*(1/(2*0.5**2)) ) )
						# Cd = np.exp( -( ((0.50*incircle_A)/rangecircle_A)*(1/(2*0.5**2)) ) )
						# Cd = 0.5*( 1 + np.tanh( 3.0*(rangecircle_A/(1.0*incircle_A)-0.5) ) )*\
						# 	0.5*( 1 + np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
						# Cd = 0.5*( 1 + np.tanh( 10.0*( (rangecircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.7) ) )
						Cd = 0.5*( 1 + np.tanh( 15.0*( (rangecircle_A/(1.0*incircle_A)) - 0.8) ) )
						Cd *= C_descent[-1]
						C_descent.append(Cd)

					Cn_.append(Cn)
					# print("Cd_4: " + str(Cd))
					# print("Cn_4: " + str(Cn))
					# print("dx_4: " + str(dx) + "\n")
				elif np.shape(target_points)[0] - i == 3:

					nodes = target_points[0:np.shape(target_points)[0]-i]
					x = [element[0] for element in nodes]; avg_x = np.mean(x)
					y = [element[1] for element in nodes]; avg_y = np.mean(y)
					
					geometric_center = np.array([avg_x, avg_y])
					rangecircle_r, R_hold = 0, 0

					for (element, i) in zip(nodes, range(len(nodes))):

						p1 = geometric_center
						p2 = np.array([element[0], element[1]])

						distance_ = np.linalg.norm(p1 - p2)

						if distance_ >= R_hold:

							rangecircle_r = distance_
							R_hold = distance_

					circumcircle_A = np.pi*(rangecircle_r)**2

					# nodes = target_points[0:np.shape(target_points)[0]-i]

					# x = [element[0] for element in nodes]; avg_x = np.mean(x)
					# y = [element[1] for element in nodes]; avg_y = np.mean(y)
					# geometric_center = np.array([avg_x, avg_y])
					
					# circumcircle_x, circumcircle_y, circumcircle_r = self.circumcenter(nodes)
					# circumcircle_A = np.pi*circumcircle_r**2
					# theta = self.calculate_tangent_angle((circumcircle_x, circumcircle_y), circumcircle_r, self.pos)


					# Cn = np.exp( -( (circumcircle_A/(0.8*incircle_A))*(1/(2*0.5**2)) ) )*\
					# 	np.exp( -( ((theta)/(1*self.alpha))*(1/(2*0.5**2)) ) )
					# Cn = np.exp( -( (circumcircle_A/(1.0*incircle_A))*(1/(2*0.5**2)) ) )
					# Cn = 0.5*( 1 - np.tanh( 3.0*(circumcircle_A/(1.0*incircle_A)-0.5) ) )*\
					# 	0.5*( 1 - np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
					# Cn = 0.5*( 1 - np.tanh( 10.0*( (circumcircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.8) ) )
					Cn = 0.5*( 1 - np.tanh( 15.0*( (circumcircle_A/(1.0*incircle_A)) - 0.8) ) )

					# print("circumcircle_x, circumcircle_y, circumcircle_r, circumcircle_A: ", end='')
					# print(str(circumcircle_x), str(circumcircle_y), str(circumcircle_r), str(circumcircle_A))
					# print("incircle_A: " + str(incircle_A))
					# print("theta: " + str(theta))
					# print("self.alpha: " + str(self.alpha))
					# print("Cn_3: " + str(Cn))

					if len(C_descent) == 0:

						# dx += (-Cn)*(-2)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])]))

						# Cd = np.exp( -( ((0.25*incircle_A)/circumcircle_A)*(1/(2*0.5**2)) ) )*\
						# 	np.exp( -( ((0.25*self.alpha)/(theta))*(1/(2*0.5**2)) ) )
						# Cd = np.exp( -( ((0.50*incircle_A)/circumcircle_A)*(1/(2*0.5**2)) ) )
						# Cd = 0.5*( 1 + np.tanh( 3.0*(circumcircle_A/(1.0*incircle_A)-0.5) ) )*\
						# 	0.5*( 1 + np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
						# Cd = 0.5*( 1 + np.tanh( 10.0*( (circumcircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.8) ) )
						Cd = 0.5*( 1 + np.tanh( 15.0*( (circumcircle_A/(1.0*incircle_A)) - 0.8) ) )
						C_descent.append(Cd)
					else:

						Cn *= C_descent[-1]
						# dx += (-Cn)*(-2)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(geometric_center[0]-self.virtual_target[0]), (geometric_center[1]-self.virtual_target[1])]))

						# Cd = np.exp( -( ((0.25*incircle_A)/circumcircle_A)*(1/(2*0.5**2)) ) )*\
						# 	np.exp( -( ((0.25*self.alpha)/(theta))*(1/(2*0.5**2)) ) )
						# Cd = np.exp( -( ((0.50*incircle_A)/circumcircle_A)*(1/(2*0.5**2)) ) )
						# Cd = 0.5*( 1 + np.tanh( 3.0*(circumcircle_A/(1.0*incircle_A)-0.5) ) )*\
						# 	0.5*( 1 + np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
						# Cd = 0.5*( 1 + np.tanh( 10.0*( (circumcircle_A/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.8) ) )
						Cd = 0.5*( 1 + np.tanh( 15.0*( (circumcircle_A/(1.0*incircle_A)) - 0.8) ) )
						Cd *= C_descent[-1]
						C_descent.append(Cd)

					Cn_.append(Cn)
					# print("Cd_3: " + str(Cd))
					# print("Cn_3: " + str(Cn))
					# print("dx_3: " + str(dx) + "\n")
				elif np.shape(target_points)[0] - i == 2:

					nodes = target_points[0:np.shape(target_points)[0]-i]

					p1 = np.array(nodes[0])
					p2 = np.array(nodes[1])
					sidecircle_center = np.array([0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1])])
					sidecircle_r = 0.5*np.linalg.norm(p1-p2); sidecircle_A = np.pi*sidecircle_r**2
					theta = self.calculate_tangent_angle(sidecircle_center, sidecircle_r, self.pos)

					# print("sidecircle_center, sidecircle_r, sidecircle_A: ", end="")
					# print(str(sidecircle_center), str(sidecircle_r), str(sidecircle_A))
					# print("theta: " + str(theta))

					# Cn = np.exp( -( ((1.0*sidecircle_A)/(incircle_A))*(1/(2*0.5**2)) ) )*\
					# 	np.exp( -( ((1.0*theta)/(self.alpha))*(1/(2*0.5**2)) ) )
					# Cn = np.exp( -( ((1.0*sidecircle_A)/(incircle_A))*(1/(2*0.5**2)) ) )
					# Cn = 0.5*( 1 - np.tanh( 3.0*((1.0*sidecircle_A)/(1.0*incircle_A)-0.5) ) )*\
					# 	0.5*( 1 - np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
					# Cn = 0.5*( 1 - np.tanh( 10.0*( ((1.0*sidecircle_A)/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.9) ) )
					Cn = 0.5*( 1 - np.tanh( 15.0*( ((1.0*sidecircle_A)/(1.0*incircle_A)) - 0.9) ) )
					# print("Cn_2: " + str(Cn))
					# C1 = np.exp( -( ((0.25*incircle_A)/sidecircle_A)*(1/(2*0.5**2)) ) )*\
					# 	np.exp( -( (0.25*self.alpha/theta)*(1/(2*0.5**2)) ) )
					# C1 = np.exp( -( ((0.50*incircle_A)/sidecircle_A)*(1/(2*0.5**2)) ) )
					# C1 = 0.5*( 1 + np.tanh( 3.0*((1.0*sidecircle_A)/(1.0*incircle_A)-0.5) ) )*\
					# 	0.5*( 1 + np.tanh( 3.0*((theta)/(1*self.alpha)-0.5) ) )
					# C1 = 0.5*( 1 + np.tanh( 10.0*( ((1.0*sidecircle_A)/(1.0*incircle_A))*((theta)/(1*self.alpha)) - 0.9) ) )
					C1 = 0.5*( 1 + np.tanh( 15.0*( ((1.0*sidecircle_A)/(1.0*incircle_A)) - 0.9) ) )
					# print("Cn_1: " + str(C1))

					if len(C_descent) == 0:

						# dx += (-Cn)*(-2)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])])+\
						# (-C1)*(-2)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])]))
						# Dx.append(-(-2)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])])+\
						(C1)*(1.0)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])]))
						# Dx.append(1.0*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])]))
					else:

						Cn *= C_descent[-1]
						C1 *= C_descent[-1]
						# dx += (-Cn)*(-2)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])])+\
						# (-C1)*(-2)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])])
						# Dx.append(-(-2)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])]))
						# Dx.append(-(-2)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])]))
						dx += (Cn)*(1.0)*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])])+\
						(C1)*(1.0)*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])])
						# Dx.append(1.0*np.array([(sidecircle_center[0]-self.virtual_target[0]), (sidecircle_center[1]-self.virtual_target[1])]))
						# Dx.append(1.0*np.array([(nodes[0][0]-self.virtual_target[0]), (nodes[0][1]-self.virtual_target[1])]))

					Cn_.append(Cn)
					Cn_.append(C1)
					# print("dx_21: " + str(dx) + "\n")
			# Cn_ = np.array(Cn_)/np.sum(np.array(Cn_))
			# # print("Cn: ", Cn_)
			# # print("dx: ", Dx)

			# dx = np.zeros(2)
			# for (i, element) in zip(range(len(Dx)), Dx):

			# 	dx += Cn_[i]*element

			# print("dx: ", dx)

		self.virtual_target += 0.8*dx
		# self.virtual_target += 0.40*dx
		# print("virtual_target: " + str(self.virtual_target) + "\n")
		self.target = [[self.virtual_target, 2.0, 10]]
		print("self_target: " + str(self.target))

		# ANOT
		# pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		# polygon = Polygon(pt)

		# Observe_list = np.zeros(len(targets))

		# for (mem, i) in zip(targets, range(len(targets))):

		# 	gemos = Point(mem[0])

		# 	if polygon.is_valid and polygon.contains(gemos):

		# 		Observe_list[i] = 1

		# Observe_list = np.append(Observe_list, time_)

	def calculate_tangent_angle(self, circle_center, circle_radius, point):

		distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)

		if (distance - circle_radius) <= 0.5 or distance <= circle_radius:

			angle = 30*(np.pi/180)
		else:

			adjcent = np.sqrt(distance**2 - circle_radius**2)
			angle = 2*np.arctan(circle_radius/adjcent)

		return angle

	def Gradient_Ascent(self, targets, time_):

		translational_force = np.array([0.,0.])
		rotational_force = np.array([0.,0.])
		zoom_force = 0.0

		W = self.W[np.where(self.FoV > 0)]; pos = self.pos; lamb = float(self.lamb); R = float(self.R); R_ = R**lamb
		alpha = self.alpha; perspective = self.perspective

		# Bivariate Normal Distribution
		F = multivariate_normal([self.target[0][0][0], self.target[0][0][1]],\
								[[self.target[0][1], 0.0], [0.0, self.target[0][1]]])
		F_ = np.array([F.pdf(W)])
		
		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)
		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		d = np.array([d]).transpose()
		d[np.where(d == 0)] = 1 # (905, 1)

		# Position
		p_dot = np.empty_like(translational_force)

		# Left Derivative
		d_ = d.transpose()[0]; const = np.empty_like(d_)
		ne.evaluate("( (lamb+1)/(1-cos(alpha)) )*( cos(alpha) - (lamb/(R*lamb+R))*d_ )*( d_**lamb/R_ )", out = const)
		hold = np.empty_like(const); const = np.array([const]).transpose()
		d_ = np.concatenate((d,d), axis = 1)
		ne.evaluate("sum(((W - pos)/(d_**3))*perspective, axis = 1)", out = hold); hold = np.array([hold]).transpose()
		der_1 = np.empty_like(d_)
		ne.evaluate("const*( hold*(W - pos) - perspective/d )", out = der_1)

		# Right Derivative
		d_ = d.transpose()[0]; const = np.empty_like(d_)
		ne.evaluate("sum(((W - pos)/d)*perspective, axis = 1)", out = const); const = np.array([const]).transpose()
		d_ = np.concatenate((d,d), axis = 1); hold = np.empty_like(d_)
		ne.evaluate("(cos(alpha)/R_)*(-lamb)*(d**(lamb-2))*(W - pos) + (lamb/(R**(lamb+1)))*(d**(lamb-1))*(W - pos)", out = hold);
		der_2 = np.empty_like(d_)
		ne.evaluate("(lamb+1)/(1-cos(alpha))*(const-cos(alpha))*hold", out = der_2)

		phi = F_.transpose()
		ne.evaluate("sum((der_1 + der_2)*phi*0.1, axis = 0)", out = p_dot)
		# print("p_dot: " + str(p_dot))
		p_norm = np.linalg.norm(p_dot)
		p_dot /= p_norm
		# p_dot = np.array([np.sign(p_dot[0]), np.sign(p_dot[1])]) + np.tanh(p_dot)
		# print("p_dot: " + str(p_dot))

		if (np.isnan(p_dot).any()):

			# print(halt)

			p_dot = np.array([0.1, 0.1])
		# 	print("p_dot: " + str(p_dot))

		# Perspective
		v_dot = np.empty_like(rotational_force)

		d_ = d.transpose()[0]; hold = np.empty_like(d_); F_ = np.array([F.pdf(W)]);
		ne.evaluate("( (lamb+1)/(1-cos(alpha)) )*( cos(alpha) - (lamb/(R*lamb+R))*d_ )*( (d_**lamb)/R_ )", out = hold)
		hold = np.array([hold]).transpose();
		d_ = np.concatenate((d,d), axis = 1); out = np.empty_like(d_)
		ne.evaluate("hold*( (W - pos)/d )", out = out)

		phi = F_.transpose();
		ne.evaluate("sum( out*phi, axis = 0)", out = v_dot)
		v_dot = v_dot/np.linalg.norm(v_dot);

		# print("v_dot: " + str(v_dot))

		# Angle of View
		a_dot = np.empty_like(zoom_force)

		# Left Derivative
		der_1 = np.empty_like(d)
		ne.evaluate("(lamb+1)*(-sin(alpha))*((d**lamb)/R_)", out = der_1)

		# Right Derivative
		const = np.empty_like(d);
		ne.evaluate("(lamb+1)*( sin(alpha)/((1-cos(alpha))**2) )*( 1 - (lamb/(R*lamb+R))*d )*( (d**lamb)/R_ )", out = const)
		d_ = d.transpose()[0]; hold = np.empty_like(d_)
		ne.evaluate("sum(((W - pos)/d)*perspective, axis = 1)", out = hold); hold = np.array([hold]).transpose()
		der_2 = np.empty_like(d);
		ne.evaluate("const*( 1 - hold )", out = der_2)

		phi = F_.transpose()
		ne.evaluate("sum( (der_1 + der_2)*phi)", out = a_dot)

		# print("a_dot: " + str(a_dot))

		if abs(a_dot) <= 0.001:

			a_dot = 0.13

		# Utiltiy Function
		H = np.empty_like(zoom_force)
		for t in targets:

			F = multivariate_normal([t[0][0], t[0][1]],\
									[[0.1, 0.0], [0.0, 0.1]])
			F_ = np.array([F.pdf(W)])
			phi = F_.transpose()

			hold = self.FoV[np.where(self.FoV > 0)]; hold = np.array([hold]).transpose()
			# print("hold: ", np.shape(hold))
			# print("phi: ", np.shape(phi))
			# ne.evaluate("sum( hold*phi )", out = H)
			H += ne.evaluate("sum( hold*phi )")

		# print("H: " + str(H))
		# print("self.H: " + str(self.H))
		# print("rate: " + str((H - self.H)/self.H))
 
		range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		dist = np.linalg.norm(self.target[0][0]-self.pos)
		dist_s = np.linalg.norm(self.target[0][0]-self.sweet_spot)

		# if (H - self.H)/self.H <= 0.025 and H > 1:
		P1_gain = min(3.0 + 3*np.exp(dist_s-3.0), 3.0)
		P2_gain = min(3.5*dist_s, 2.5)
		# P_gain = min(0.12*np.exp(dist_s), 2.3)
		P_gain = min(0.25*np.exp(dist_s), 2.3)
		# print("UAV id, P1_gain:", self.id, P1_gain)
		# print("UAV id, P2_gain:", self.id, P2_gain)

		if dist <= range_max:

			# self.translational_force = 0.1*np.tanh(p_norm)*p_dot
			# self.translational_force = 1.0*P2_gain*p_dot
			self.translational_force = 1.0*P_gain*p_dot
			# self.translational_force = 0.08*p_dot
			self.perspective_force = 60*np.asarray([v_dot[0], v_dot[1]])
			self.zoom_force = 0.01*a_dot
			self.stage = 2
			self.r = self.R*cos(self.alpha)
		else:

			# self.translational_force = 3*p_dot
			# self.translational_force = P1_gain*p_dot
			self.translational_force = 1.0*P_gain*p_dot
			self.perspective_force = 60*np.asarray([v_dot[0], v_dot[1]])
			self.zoom_force = 0.01*a_dot
			self.stage = 2
			self.r = self.R*cos(self.alpha)

		print("translational_force: ", self.translational_force)

		self.H = H
		T_Utility = [self.H, time_]

	def PublishMessage(self):

		h_dim = MultiArrayDimension(label = "height", size = 1, stride = 1*1*3)
		w_dim = MultiArrayDimension(label = "width",  size = 1, stride = 1*1)
		c_dim = MultiArrayDimension(label = "channel", size = 3, stride = 3)
		layout = MultiArrayLayout(dim = [h_dim, w_dim, c_dim], data_offset = 0)

		curr_vel = [self.translational_force[0], self.translational_force[1]]
		output_ = Float64MultiArray(data = curr_vel, layout = layout)
		self.Curr_vel_pub.publish(output_)

		if len(self.comc[str(self.one_hop_neighbor[0])]) < 1:
		
			target_residual = []
		else:

			target_residual = []
			print("self.comc: ", self.comc[str(self.one_hop_neighbor[0])])
			for element in self.comc[str(self.one_hop_neighbor[0])]:

				print("element: ", )

				target_residual.extend(element)

		output_ = Float64MultiArray(data = target_residual, layout = layout)
		self.Target_Resdiual_pub.publish(output_)

		output_ = Float64MultiArray(data = self.sweet_spot, layout = layout)
		self.Sweet_Spot_pub.publish(output_)

		target3d_z = np.array(self.targets_3dpos)[self.cluster_set]
		z = np.sum(target3d_z[:,2])/len(target3d_z)
		# print("target3d_z: ", target3d_z)
		# print("z: ", z)
		output_ = Odometry()
		output_.pose.pose.position.x = self.target[0][0][0]
		output_.pose.pose.position.y = self.target[0][0][1]
		output_.pose.pose.position.z = z
		output_.twist.twist.linear.x = 0.0
		output_.twist.twist.linear.y = 0.0
		output_.twist.twist.linear.z = 0.0
		self.target_pub.publish(output_)

	def UpdateFoV(self):

		W = self.W; pos = self.pos; perspective = self.perspective; alpha = self.alpha; R = self.R
		lamb = self.lamb; R_ = R**(lamb+1)

		out = np.empty_like(self.W)
		ne.evaluate("W - pos", out = out)

		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		# d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d = d.transpose()[0]

		q_per = self.PerspectiveQuality(d, W, pos, perspective, alpha)
		q_res = self.ResolutionQuality(d, W, pos, perspective, alpha, R, lamb)
		Q = np.multiply(q_per, q_res)

		# print("Perspective: " + str(q_per))
		# print("Resolution: " + str(q_res))
		# print(halt)

		quality_map = ne.evaluate("where((q_per > 0) & (q_res > 0), Q, 0)")
		self.FoV = quality_map
		
		return

	def PerspectiveQuality(self, d, W, pos, perspective, alpha):

		out = np.empty_like(d)
		ne.evaluate("sum((W - pos)*perspective, axis = 1)", out = out)
		ne.evaluate("(out/d - cos(alpha))/(1 - cos(alpha) )", out = out)

		# return (np.divide(np.dot(np.subtract(self.W, self.pos), self.perspective), d) - np.cos(self.alpha))\
		# 		/(1 - np.cos(self.alpha))
		return out

	def ResolutionQuality(self, d, W, pos, perspective, alpha, R, lamb):

		R_ = R**(lamb+1)

		out = np.empty_like(d)
		ne.evaluate("(R*cos(alpha) - lamb*(d - R*cos(alpha)))*(d**lamb)/R_", out = out)

		# return np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
		# 					(np.power(d, self.lamb)/(self.R**(self.lamb+1))))
		return out

	def UpdateLocalVoronoi(self):

		id_ = self.id
		quality_map = self.FoV

		# for neighbor in self.neighbors:

		# 	FoV = neighbor.FoV
		# 	quality_map = ne.evaluate("where((quality_map >= FoV), quality_map, 0)")

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV > 0)))
		# self.map_plt = np.array(ne.evaluate("where(quality_map != 0, id_ + 1, 0)"))

		return

	def FormationControl(self, targets):

		# Replusive Force of Targets -----------------------------------------------------------------------------------------------
		tracker_margin = 1.2
		enemy_force = np.zeros(2)

		if len(targets) > 1:

			for k in range(len(targets)):

				dist = np.linalg.norm(self.pos - targets[k][0])

				if dist < tracker_margin:

					# Adjust velocities to avoid collision
					direction = np.array((self.pos - targets[k][0])/np.linalg.norm(self.pos - targets[k][0]))
					heading = np.array((self.pos - self.target[0][0])/np.linalg.norm(self.pos - self.target[0][0]))
					theta = np.arccos(np.dot(direction, heading)); trun = (direction*np.sin(theta))/np.linalg.norm(direction*np.sin(theta))
					enemy_force += +1.5*(tracker_margin - dist)*trun
					# direction = (self.pos - targets[k][0])
					# enemy_force += +(tracker_margin - dist)*(direction/np.linalg.norm(direction))

			if (enemy_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm

				enemy_force_norm = np.linalg.norm(enemy_force)
				enemy_force_normal = enemy_force/enemy_force_norm
				# print("enemy_force: ", enemy_force)

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, enemy_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, enemy_force_normal))

				enemy_force_transvers = enemy_force_norm*np.cos(theta_)*center_force_R_normal
				enemy_force_transvers_norm = np.linalg.norm(enemy_force_transvers)
				enemy_force_transvers_normal = enemy_force_transvers/enemy_force_transvers_norm
				# enemy_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(enemy_force)*np.sin(theta_)
			else:

				enemy_force_transvers = np.zeros(2)
		else:

				enemy_force_transvers = np.zeros(2)

		# Neighbors Force---------------------------------------------------------------------------------------------------------
		neighbors_pos = []; neighbors_pos.append(self.pos)

		if len(self.neighbors["Position"]) > 0:

			neighbors_pos = np.concatenate((neighbors_pos, [value for key, value in self.neighbors["Position"].items()]))

			# Calculate the pairwise distances between targets
			distances = distance.cdist(neighbors_pos, neighbors_pos)[0]
			distances = np.delete(distances, 0)
			weight = 1 - 0.5*( 1 + np.tanh(2*(distances-5)) )

			# Consider which neighbor is sharing the same target and only use them to obtain formation force
			neighbor_force = 0.0

			# for (i, neighbor) in zip(range(len(distances)), self.neighbors["Position"]):

			# 	neighbor_force += weight[i]*((self.pos - neighbor_pos)/(np.linalg.norm(self.pos - neighbor_pos)))
			# for (i, key) in zip(range(len(distances)), self.neighbors["Position"]):

			# 	neighbor_force += weight[i]*((self.pos - self.neighbors["Position"][key])/(np.linalg.norm(self.pos - self.neighbors["Position"][key])))
			for key in self.one_hop_neighbor:

				neighbor_force += 10.0*((self.pos - self.neighbors["Position"][str(key)])\
										/(np.linalg.norm(self.pos - self.neighbors["Position"][str(key)])))

			if (neighbor_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm

				neighbor_force_norm = np.linalg.norm(neighbor_force)
				neighbor_force_normal = neighbor_force/neighbor_force_norm
				# print("neighbor_force: ", neighbor_force)

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, neighbor_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, neighbor_force_normal))

				neighbor_force_transvers = neighbor_force_norm*np.cos(theta_)*center_force_R_normal
				neighbor_force_transvers_norm = np.linalg.norm(neighbor_force_transvers)
				neighbor_force_transvers_normal = neighbor_force_transvers/neighbor_force_transvers_norm
				# neighbor_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(neighbor_force)*np.sin(theta_)
			else:

				neighbor_force_transvers = np.zeros(2)
		else:

			neighbor_force_transvers = np.zeros(2)

		# Herding Force ------------------------------------------------------------------------------------------------------------
		if (np.isnan(self.Pd[0]).any()):

			herd_force_transvers = np.zeros(2)
		else:

			Pc_force = self.Pd - self.pos
			Pc_force_norm = np.linalg.norm(Pc_force)
			Pc_force_normal = Pc_force/Pc_force_norm

			center_force = (np.asarray(self.target[0][0]) - self.pos)
			center_force_norm = np.linalg.norm(center_force)
			center_force_normal = center_force/center_force_norm

			theta_alert = np.arccos(np.dot(center_force_normal, Pc_force_normal))
			if theta_alert <= 2*(np.pi/180) and Pc_force_norm > center_force_norm:

				herd_force_transvers = enemy_force_transvers + neighbor_force_transvers
				herd_force_transvers_norm = np.linalg.norm(herd_force_transvers)
				herd_force_transvers_normal = herd_force_transvers/herd_force_transvers_norm
			else:

				herd_force = (np.array(self.Pd) - self.pos)
				herd_force_norm = np.linalg.norm(herd_force)
				herd_force_normal = herd_force/herd_force_norm

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, herd_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, herd_force_normal))

				herd_force_transvers = 1.5*herd_force_norm*np.cos(theta_)*center_force_R_normal
				herd_force_transvers_norm = np.linalg.norm(herd_force_transvers)
				herd_force_transvers_normal = herd_force_transvers/herd_force_transvers_norm

				# herding_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(np.array(self.Pd) - self.pos)*np.sin(theta_)

		herd_force_transvers = np.zeros(2)
		# Formation Control-------------------------------------------------------------------------------------------------------
		original_translation_force = self.translational_force
		# herd_force_transvers = np.zeros(2)
		if self.stage == 2:

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

			center_force_norm = np.linalg.norm(np.asarray(self.target[0][0]) - self.pos)

			repulsive_force = (self.pos - np.asarray(self.target[0][0]))\
				/(np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			repulsive_force_norm = np.linalg.norm(self.pos - np.asarray(self.target[0][0]))

			# drive_force = herd_force_transvers
			drive_force = enemy_force_transvers + neighbor_force_transvers + herd_force_transvers
			drive_force_norm = np.linalg.norm(drive_force)

			formation_force = (center_force*(drive_force_norm/(center_force_norm + drive_force_norm))\
							+ drive_force*(center_force_norm/(center_force_norm + drive_force_norm)))

			formation_force += repulsive_force*(self.r - np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			self.translational_force += (formation_force)
		else:

			# formation_force = herd_force_transvers
			formation_force = (enemy_force_transvers + neighbor_force_transvers + herd_force_transvers)
			self.translational_force += formation_force

		# print("Herding Force: ", herd_force_transvers)

	def CBF_Collision_Herding(self, targets):

		# Replusive Force of Targets -----------------------------------------------------------------------------------------------
		tracker_margin = 1.5
		enemy_force = np.zeros(2)

		if len(targets) > 1:

			for k in range(len(targets)):

				dist = np.linalg.norm(self.pos - targets[k][0])

				if dist < tracker_margin:

					# Adjust velocities to avoid collision
					direction = np.array((self.pos - targets[k][0])/np.linalg.norm(self.pos - targets[k][0]))
					heading = np.array((self.pos - self.target[0][0])/np.linalg.norm(self.pos - self.target[0][0]))
					theta = np.arccos(np.dot(direction, heading)); trun = (direction*np.sin(theta))/np.linalg.norm(direction*np.sin(theta))
					enemy_force += +1.5*(tracker_margin - dist)*trun
					# direction = (self.pos - targets[k][0])
					# enemy_force += +(tracker_margin - dist)*(direction/np.linalg.norm(direction))

			if (enemy_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm

				enemy_force_norm = np.linalg.norm(enemy_force)
				enemy_force_normal = enemy_force/enemy_force_norm
				# print("enemy_force: ", enemy_force)

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, enemy_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, enemy_force_normal))

				enemy_force_transvers = enemy_force_norm*np.cos(theta_)*center_force_R_normal
				enemy_force_transvers_norm = np.linalg.norm(enemy_force_transvers)
				enemy_force_transvers_normal = enemy_force_transvers/enemy_force_transvers_norm
				# enemy_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(enemy_force)*np.sin(theta_)
			else:

				enemy_force_transvers = np.zeros(2)
		else:

				enemy_force_transvers = np.zeros(2)

		# Neighbors Force---------------------------------------------------------------------------------------------------------
		neighbors_pos = []; neighbors_pos.append(self.pos)

		if len(self.neighbors["Position"]) > 0:

			neighbors_pos = np.concatenate((neighbors_pos, [value for key, value in self.neighbors["Position"].items()]))

			# Calculate the pairwise distances between targets
			distances = distance.cdist(neighbors_pos, neighbors_pos)[0]
			distances = np.delete(distances, 0)
			weight = 1 - 0.5*( 1 + np.tanh(2*(distances-5)) )

			# Consider which neighbor is sharing the same target and only use them to obtain formation force
			neighbor_force = 0.0

			# for (i, neighbor) in zip(range(len(distances)), self.neighbors["Position"]):

			# 	neighbor_force += weight[i]*((self.pos - neighbor_pos)/(np.linalg.norm(self.pos - neighbor_pos)))

			if len(self.teammate) == 1:

				for (i, key) in zip(range(len(distances)), self.neighbors["Position"]):

					neighbor_force += weight[i]*((self.pos - self.neighbors["Position"][key])/(np.linalg.norm(self.pos - self.neighbors["Position"][key])))
			else:
				# for key in self.one_hop_neighbor:

				# 	neighbor_force += 10.0*((self.pos - self.neighbors["Position"][str(key)])\
				# 							/(np.linalg.norm(self.pos - self.neighbors["Position"][str(key)])))
				for (i, key) in zip(range(len(distances)), self.neighbors["Position"]):

					neighbor_force += weight[i]*((self.pos - self.neighbors["Position"][key])/(np.linalg.norm(self.pos - self.neighbors["Position"][key])))

			if (neighbor_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm

				neighbor_force_norm = np.linalg.norm(neighbor_force)
				neighbor_force_normal = neighbor_force/neighbor_force_norm
				# print("neighbor_force: ", neighbor_force)

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, neighbor_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, neighbor_force_normal))

				neighbor_force_transvers = neighbor_force_norm*np.cos(theta_)*center_force_R_normal
				neighbor_force_transvers_norm = np.linalg.norm(neighbor_force_transvers)
				neighbor_force_transvers_normal = neighbor_force_transvers/neighbor_force_transvers_norm
				# neighbor_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(neighbor_force)*np.sin(theta_)
			else:

				neighbor_force_transvers = np.zeros(2)
		else:

			neighbor_force_transvers = np.zeros(2)

		# Herding Force ------------------------------------------------------------------------------------------------------------
		# if (np.isnan(self.Pd[0]).any()):
		if np.array_equal(self.Pd, np.zeros(2)):

			herd_force_transvers = np.zeros(2)
		else:

			Pc_force = self.Pd - self.pos
			Pc_force_norm = np.linalg.norm(Pc_force)
			Pc_force_normal = Pc_force/Pc_force_norm

			center_force = (np.asarray(self.target[0][0]) - self.pos)
			center_force_norm = np.linalg.norm(center_force)
			center_force_normal = center_force/center_force_norm

			theta_alert = np.arccos(np.dot(center_force_normal, Pc_force_normal))
			if theta_alert <= 2*(np.pi/180) and Pc_force_norm > center_force_norm:

				herd_force_transvers = enemy_force_transvers + neighbor_force_transvers
				herd_force_transvers_norm = np.linalg.norm(herd_force_transvers)
				herd_force_transvers_normal = herd_force_transvers/herd_force_transvers_norm
			else:

				herd_force = (np.array(self.Pd) - self.pos)
				herd_force_norm = np.linalg.norm(herd_force)
				herd_force_normal = herd_force/herd_force_norm

				# Rotating Direction
				direction_ = np.sign(np.cross(center_force_normal, herd_force_normal))
				R = np.array([[np.cos(direction_*np.pi*0.5), -np.sin(direction_*np.pi*0.5)],
								[np.sin(direction_*np.pi*0.5), np.cos(direction_*np.pi*0.5)]])

				# Rotating Magnitude
				center_force_R = np.dot(R, center_force)
				center_force_R_norm = np.linalg.norm(center_force_R)
				center_force_R_normal = center_force_R/center_force_R_norm

				theta_ = np.arccos(np.dot(center_force_R_normal, herd_force_normal))

				# herd_force_transvers = 1.5*herd_force_norm*np.cos(theta_)*center_force_R_normal
				herd_force_transvers = 2.0*herd_force_norm*np.cos(theta_)*center_force_R_normal
				herd_force_transvers_norm = np.linalg.norm(herd_force_transvers)
				herd_force_transvers_normal = herd_force_transvers/herd_force_transvers_norm

				# herding_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(np.array(self.Pd) - self.pos)*np.sin(theta_)

		# if len(self.teammate) > 1:

		# 	herd_force_transvers = np.zeros(2)
		# print("UAV id, Pd: ", self.id, self.Pd)
		# print("UAV id, herd_force_transvers: ", self.id, herd_force_transvers)

		# Formation Control-------------------------------------------------------------------------------------------------------
		original_translation_force = self.translational_force

		if self.stage == 2:

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

			center_force_norm = np.linalg.norm(np.asarray(self.target[0][0]) - self.pos)

			repulsive_force = (self.pos - np.asarray(self.target[0][0]))\
				/(np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			repulsive_force_norm = np.linalg.norm(self.pos - np.asarray(self.target[0][0]))

			# drive_force = herd_force_transvers
			drive_force = enemy_force_transvers + neighbor_force_transvers + herd_force_transvers
			drive_force_norm = np.linalg.norm(drive_force)

			formation_force = (center_force*(drive_force_norm/(center_force_norm + drive_force_norm))\
							+ drive_force*(center_force_norm/(center_force_norm + drive_force_norm)))

			formation_force += repulsive_force*(self.r - np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			self.translational_force += (formation_force)
		else:

			# formation_force = herd_force_transvers
			formation_force = enemy_force_transvers + herd_force_transvers
			self.translational_force += formation_force


		# CBF - Collision
		# communication_range = 2.0*self.incircle_r
		# dt_min = 0.2*self.incircle_r
		# da_min = 0.2*self.incircle_r
		communication_range = 3.0
		dt_min = 1.0
		da_min = 1.0
		G, H = [], []

		for i in range(len(targets)):

			dist = np.linalg.norm(self.pos - targets[i][0])

			if dist < communication_range:

				d = (self.pos - targets[i][0])
				h = 1.1*(dist**2 - dt_min**2) - 2*np.dot(d, self.targets_vel[i])
				G.append([-2*d[0], -2*d[1]])
				H.append(h)

		# for (i, neighbor_pos) in zip(range(len(self.neighbors["Vel"])) ,self.neighbors["Position"]):
		for (key_vel, key_pos) in zip(self.neighbors["Vel"], self.neighbors["Position"]):

			dist = np.linalg.norm(self.pos - self.neighbors["Position"][key_pos])

			if dist < communication_range:

				d = (self.pos - self.neighbors["Position"][key_vel])
				h = 1.1*(dist**2 - da_min**2) - 2*np.dot(d, self.neighbors["Vel"][key_vel])
				G.append([-2*d[0], -2*d[1]])
				H.append(h)

		# Herding
		if len(self.teammate) == 1:

			theta_max = 25*(np.pi/180)
			V = (self.C-self.GCM)/np.linalg.norm(self.C-self.GCM)
			L = np.dot((self.pos-self.C)/np.linalg.norm((self.pos-self.C)), V)
			h_theta = theta_max - np.arccos(L)

			if np.arccos(L) < theta_max:

				A_2 = ((-1)/np.sqrt(1-L**2))*( V/np.linalg.norm((self.pos-self.C)) - \
									(np.dot((self.pos-self.C), V)/np.linalg.norm(self.pos-self.C)**3)*(self.pos-self.C) )

				G.append(A_2.tolist())
				H.append(h_theta)

		# print("G: ", G)
		# print("H: ", H)

		if len(G) > 0 and len(H) > 0:

			if len(G) == 1:

				# QP
				Q = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
				p = matrix([-2*self.translational_force[0], -2*self.translational_force[1]])
				G = matrix(G, (1,2))
				# if min(G) > 1:

					# G /= max(G)

				H = matrix(H)
				# if min(H) > 1:

				# 	H /= max(H)

				# print("Q: ", Q)
				# print("p: ", p)
				# print("G: ", G)
				# print("H: ", H)
			else:

				# QP
				Q = 2*matrix([[1.0, 0.0],[0.0, 1.0]])
				p = matrix([-2*self.translational_force[0], -2*self.translational_force[1]])
				G = matrix(np.transpose(G))
				G = G.trans()
				# if min(G) > 1:

					# G /= max(G)
				H = matrix(H)
				# if min(H) > 1:

					# H /= max(H)

				# print("Q: ", Q)
				# print("p: ", p)
				# print("G: ", G)
				# print("H: ", H)

			try:

				sol = solvers.coneqp(Q, p, G, H)
				control_input = np.array([sol["x"][0], sol["x"][1]])
				control_input_norm = np.linalg.norm(control_input)
				control_input_norz = control_input/control_input_norm
				gain = min(control_input_norm, 3.0)

				self.translational_force = gain*control_input_norz
				# print("solve: ", sol["x"], "\n")
			except:

				if self.stage == 2:

					center_force = (np.asarray(self.target[0][0]) - self.pos)\
						/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

					center_force_norm = np.linalg.norm(np.asarray(self.target[0][0]) - self.pos)

					repulsive_force = (self.pos - np.asarray(self.target[0][0]))\
						/(np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

					repulsive_force_norm = np.linalg.norm(self.pos - np.asarray(self.target[0][0]))

					drive_force = enemy_force_transvers + neighbor_force_transvers + herd_force_transvers
					# drive_force = enemy_force_transvers + neighbor_force_transvers
					drive_force_norm = np.linalg.norm(drive_force)

					formation_force = (center_force*(drive_force_norm/(center_force_norm + drive_force_norm))\
									+ drive_force*(center_force_norm/(center_force_norm + drive_force_norm)))

					formation_force += repulsive_force*(self.r - self.linalg.norm(self.pos - np.asarray(self.target[0][0])))
					self.translational_force = original_translation_force + (formation_force)

				else:
					formation_force = (enemy_force_transvers + neighbor_force_transvers + herd_force_transvers)
					# formation_force = (enemy_force_transvers + neighbor_force_transvers)
					self.translational_force = original_translation_force + formation_force


		# Nominal Control-------------------------------------------------------------------------------------------------------
		# if self.stage == 2:

		# 	repulsive_force = (self.pos - np.asarray(self.target[0][0]))\
		# 		/(np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

		# 	repulsive_force_norm = np.linalg.norm(self.pos - np.asarray(self.target[0][0]))

		# 	formation_force = repulsive_force*(self.r - self.norm(self.pos - np.asarray(self.target[0][0])))

		# 	self.translational_force += (formation_force)
		# else:

		# 	formation_force = np.array([0,0])
		# 	self.translational_force += formation_force

	def UpdateOrientation(self):

		self.perspective += self.perspective_force*self.step
		self.perspective /= np.linalg.norm(self.perspective)

		return

	def UpdateZoomLevel(self):

		if (self.alpha + self.zoom_force*self.step)  <= 20*(np.pi/180):

			self.alpha += self.zoom_force*self.step

		return

	def UpdatePosition(self):

		if self.stage == 1:

			self.pos += self.translational_force*self.step
		elif self.stage == 2:

			self.pos += self.translational_force*0.05

		return

	def UpdateSweetSpot(self):

		self.sweet_spot = self.pos + self.perspective*self.R*np.cos(self.alpha)

		return

	def polygon_FOV(self):

		range_max = (self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		R = np.array([[np.cos(self.alpha), -np.sin(self.alpha)]
					,[np.sin(self.alpha), np.cos(self.alpha)]])

		self.top = self.pos + range_max*self.perspective

		self.ltop = self.pos + range_max*np.reshape(R@np.reshape(self.perspective,(2,1)),(1,2))
		self.ltop = self.ltop[0]

		self.rtop = self.pos + range_max*np.reshape(np.linalg.inv(R)@np.reshape(self.perspective,(2,1)),(1,2))
		self.rtop = self.rtop[0]

# if __name__ == '__main__':

# 	try:
# 		rospy.init_node('controller_1')
# 		rate = rospy.Rate(100)

# 		map_size = np.array([25, 25])
# 		grid_size = np.array([0.1, 0.1])

# 		camera0 = { 'id'            :  0,
# 					'position'      :  np.array([1., 8.]),
# 					'perspective'   :  np.array([0.9,1]),
# 					'AngleofView'   :  20,
# 					'range_limit'   :  5,
# 					'lambda'        :  2,
# 					'color'         : (200, 0, 0)}

# 		uav_1 = UAV(camera0, map_size, grid_size)

# 		while uav_1.b is None:

# 			rate.sleep()

# 		uav_1.qp_ini()

# 		last = time()

# 		while not rospy.is_shutdown():

# 			uav_1.UpdateState(np.round(time() - last, 2))
# 			rate.sleep()

# 	except rospy.ROSInterruptException:
# 		pass
