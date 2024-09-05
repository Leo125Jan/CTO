import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
import copy
import random
import numpy as np
import numexpr as ne
import skfuzzy as fuzz
from scipy import optimize
from time import sleep, time
from gudhi import AlphaComplex
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.integrate import quad
from scipy import ndimage, sparse
from shapely.geometry import Point
from collections import namedtuple
from scipy.optimize import linprog
from scipy.spatial import distance
from cvxopt import matrix, solvers
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.cluster import MeanShift
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points
from scipy.sparse.csgraph import minimum_spanning_tree

solvers.options['show_progress'] = False

class PTZcon():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 50, Ka = 3, Kp = 2, step = 0.1):

		# Environment
		self.grid_size = grid_size
		self.map_size = map_size
		self.size = (int(map_size[0]/grid_size[0]), int(map_size[1]/grid_size[1]))

		x_range = np.arange(0, self.map_size[0], self.grid_size[0])
		y_range = np.arange(0, self.map_size[1], self.grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		self.W = W.transpose()
		
		# Properties of UAV
		self.id = properties['id']
		self.pos = properties['position']
		self.last_pos = properties['position']
		self.perspective = properties['perspective']/np.linalg.norm(properties['perspective'])
		self.alpha = properties['AngleofView']/180*np.pi
		self.R = properties['range_limit']
		self.lamb = properties['lambda']
		self.color = properties['color']
		self.R_max = (self.lamb + 1)/(self.lamb)*self.R*np.cos(self.alpha)
		self.r = 0
		self.top = 0
		self.ltop = 0
		self.rtop = 0
		self.H = 0
		self.centroid = None
		self.sweet_spot = self.pos + self.perspective*self.R*np.cos(self.alpha)
		self.child = np.array([None, None])

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
		self.Jx = None
		self.Winbid = None
		self.translational_force = np.array([0,0])
		self.ANOT = None

		# Tracking Configuration
		self.cluster_count = 0
		self.dist_to_cluster = np.array([0.0, 0.0, 0.0])
		self.dist_to_targets = np.array([0.0, 0.0, 0.0])
		self.Clsuter_Checklist = None
		self.state_machine = {"self": None, "mode": None, "target": None}
		self.attract_center = [3, None, 2, None, 1, None] # "0": 3, "1": 0, "2": 2, "3": 0, "4",: 1, "5": 0
		self.incircle = []

		# Relative Control Law
		self.translation_force = 0  # dynamics of positional changes
		self.perspective_force = 0  # dynamics of changing perspective direction
		self.stage = 1              # 1: Tracker, 2: Formation Cooperative
		self.target = None
		self.virtual_target = self.R*cos(self.alpha)*self.perspective
		self.target_assigned = -1
		self.step = step
		self.FoV = np.zeros(np.shape(self.W)[0])
		self.Kv = Kv                # control gain for perspective control law toward voronoi cell
		self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
		self.Kp = Kp                # control gain for positional change toward voronoi cell 
		self.event = np.zeros((self.size[0], self.size[1]))

	def UpdateState(self, targets, neighbors, one_hop_neighbor,
					Pd, cluster_set, col_ind, cluster_labels, cluster_centers, Team, target_velocity,
					time_, cp, speed_gain, sensing_range, save_type, Times):

		self.cp = cp
		self.speed_gain = speed_gain
		self.sensing_range = sensing_range
		self.save_type = save_type
		self.times = Times
		self.last_FOV = None

		if one_hop_neighbor is None:

			self.one_hop_neighbor = None
		else:
			self.one_hop_neighbor = np.array(one_hop_neighbor[self.id])[np.array(one_hop_neighbor[self.id]) != self.id]
		
		print("id: " + str(self.id))

		self.neighbors = neighbors
		self.time = time_
		self.teammate = Team

		self.UpdateFoV()
		self.polygon_FOV()
		# self.EscapeDensity(targets, time_)
		self.UpdateLocalVoronoi()

		# self.Cluster_Formation(targets)
		# self.Cluster_Assignment(targets, time_)

		self.CBAA(targets, cluster_set, col_ind)
		
		if self.cp == "PA":

			self.Gradient_Descent(targets, Pd, cluster_set, col_ind, cluster_labels, time_)
		elif self.cp == "OTO":

			self.comparsion(targets)
		elif self.cp == "HC":

			self.Hill_Climbing(targets, time_)
		elif self.cp == "K":

			self.Kmeans(targets, cluster_centers, cluster_labels, time_)
		elif self.cp == "FCM":

			self.FuzzyCMeans(targets, cluster_centers, cluster_labels, time_)
		elif self.cp == "DBSCAN":

			self.DBSCAN(targets, cluster_centers, cluster_labels, time_)

		self.Gradient_Ascent(targets, time_)
		
		# event = np.zeros((self.size[0], self.size[1]))
		# self.event = self.event_density(event, self.target, self.grid_size)
		
		# self.ComputeCentroidal(time_)
		# self.StageAssignment()
		# self.JointProbability(targets, time_)
		# self.FormationControl(targets)
		self.CBF_Collision_Herding(targets, target_velocity)
		self.UpdateOrientation()
		self.UpdateZoomLevel()
		self.UpdatePosition()
		self.UpdateSweetSpot()

		print("id: " + str(self.id), "\n")
		
	def norm(self, arr):

		sum = 0

		for i in range(len(arr)):

			sum += arr[i]**2

		return sqrt(sum)

	def event_density(self, event, target, grid_size):

		x = np.arange(event.shape[0])*grid_size[0]
 
		for y_map in range(0, event.shape[1]):

			y = y_map*grid_size[1]
			density = 0

			for i in range(len(target)):

				density += target[i][2]*np.exp(-target[i][1]*np.linalg.norm(np.array([x,y], dtype=object)\
											-np.array((target[i][0][1],target[i][0][0]))))

			event[:][y_map] = density

		return 0 + event

	def Gaussian_Normal_1D(self, x, mu, sigma):

		return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2/(2*sigma**2))

	def Cluster_Formation(self, targets, d):

		checklist = np.zeros((len(targets), len(targets)))
		threshold = d
		self.cluster_count = 0

		for i in range(len(targets)):

			for j in range(len(targets)):

				if j != i:

					p1 = np.array([targets[i][0][0], targets[i][0][1]])
					p2 = np.array([targets[j][0][0], targets[j][0][1]])

					dist = np.linalg.norm(p1 - p2)

					if dist <= threshold:

						checklist[i][j] = 1
						self.cluster_count += 1
					else:

						checklist[i][j] = 0

		self.Clsuter_Checklist = checklist

		return

	def CBAA(self, targets, cluster_set, col_ind):

		if (self.one_hop_neighbor is not None) and (self.teammate is not None):

			print("one_hop_neighbor: ", self.one_hop_neighbor)
			print("Teammate: ", self.teammate[self.id])

			if (len(self.teammate[self.id]) == 1):

				self.Jx = None
			else:

				# Auction Process
				targets_position = np.array([target[0] for target in targets])
				# print("targets_position: ", targets_position)

				# print("col_ind: ", col_ind)
				# print("cluster_set: ", cluster_set)

				task = targets_position[cluster_set[col_ind[self.id]]]
				# print("task: ", task)

				c = np.linalg.norm(task-self.pos, axis=1)
				# print("c: ", c)

				# Consensus Process
				count = 0
				for neighbor in self.neighbors:

					if (neighbor.id in self.one_hop_neighbor) and (neighbor.Winbid is None):

						count += 1
					elif (neighbor.id in self.one_hop_neighbor) and (neighbor.Winbid is not None):

						if (len(neighbor.Winbid) != len(c)):

							count += 1

				if count > 0:

					self.Jx = None
				else:

					# for neighbor in self.neighbors:

					# 	if (neighbor.id in self.one_hop_neighbor[self.id]):

					# 		print("n_id, Winbid: ", neighbor.id, neighbor.Winbid)

					win_bid = np.array([neighbor.Winbid for neighbor in self.neighbors if neighbor.id in self.one_hop_neighbor])
					print("win_bid: ", win_bid)

					cost_matrix = np.vstack((c, win_bid))
					# print("cost_matrix: ", cost_matrix)

					row, col = linear_sum_assignment(cost_matrix)
					# print("col: ", col)

					# Moderate
					n = len(cost_matrix)

					sequence_num = list(range(0, n))
					missing_numbers = [num for num in sequence_num if num not in row]
					missing_numbers_hold = missing_numbers
					# print("missing_number: ", missing_numbers)

					col_sol = {str(i): [] for i in range(n)}

					for (row, col) in zip(row, col):

						col_sol[str(row)] = col
					# print("col_sol: ", col_sol)

					while len(missing_numbers) != 0:

						cost_matrix_missing = np.array(cost_matrix[missing_numbers])
						# print("cost_matrix_missing: ", cost_matrix_missing)

						row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
						# print("row_ind_missing: ", row_ind_missing)
						# print("col_ind_missing: ", col_ind_missing)

						for (row, col) in zip(missing_numbers, col_ind_missing):

							col_sol[str(row)] = col
							missing_numbers_hold.remove(row)

						
						missing_numbers = missing_numbers_hold
					# print("col_sol: ", col_sol)

					col_ind = np.zeros(n)
					for key_, value_ in col_sol.items():

						col_ind[int(key_)] = value_
					print("col_ind: ", col_ind)

					self.Jx = int(col_ind[0])

				print("consensus Jx: ", self.Jx)

				# Winning Bid Publish
				self.Winbid = c
				# print("self.Winbid: ", self.Winbid)
	
	def Graph_Construction(self, I_vertex):

		# print("I_vertex: ", I_vertex)
		I_vertex = np.array(I_vertex, dtype=np.float64)

		if np.shape(I_vertex)[0] > 1:

			start_vertex = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
			# start_vertex = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.sweet_spot, axis=2))
			# print("self.pos: ", self.pos)
			# print("dist: ", np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))

			# print("Jx: ", self.Jx)
			if not np.all(self.Jx == None):

				start_vertex = self.Jx
			# print("start_vertex: " + str(start_vertex))

			edges = self.Hamiltonian_Path(I_vertex, start_vertex)
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

			# if len(edges) < 1:
 
			# 	trunk_hold = [[-1]]
			# 	path = [I_vertex[start_vertex]]
			# else:

			# 	trunk_hold = edges
			# 	path = I_vertex
		# if np.shape(I_vertex)[0] > 1:

		# 	start_vertex = np.argmin(np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
		# 	# print("self.pos: ", self.pos)
		# 	# print("dist: ", np.linalg.norm(I_vertex[:, np.newaxis] - self.pos, axis=2))
		# 	# print("start_vertex: " + str(start_vertex))
		# 	edges = self.Hamiltonian_Path(I_vertex, start_vertex)

		# 	trunk_hold = edges[0]
		# 	path = I_vertex
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

				for neighbor in self.neighbors:

					if (neighbor.id in self.one_hop_neighbor):

						neighbor_pos[str(neighbor.id)] = neighbor.sweet_spot

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

			# for index_ in range(1, len(nodes)):

			# 	# print("index_: ", index_)

			# 	x = [nodes[i][0] for i in range(0, index_+1)]; avg_x = np.mean(x)
			# 	y = [nodes[i][1] for i in range(0, index_+1)]; avg_y = np.mean(y)
			# 	geometric_center = np.array([avg_x, avg_y])
			# 	rangecircle_r, R = 0, 0

			# 	for i in range(1, index_+1):

			# 		p1 = geometric_center
			# 		p2 = np.array([nodes[i][0], nodes[i][1]])

			# 		distance_ = np.linalg.norm(p1 - p2)

			# 		if distance_ >= R:

			# 			rangecircle_r = distance_

			# 		# print("rangecircle_r: ", rangecircle_r)

			# 	rangecircle_A = np.pi*(rangecircle_r)**2
			# 	# print("rangecircle_A: ", rangecircle_A)
			# 	# print("incircle_r: ", self.incircle[1])
			# 	# print("incircle_A: ", self.incircle[2])

			# 	if rangecircle_A >= 1.0*self.incircle[2]:

			# 		break
			# 	else:

			# 		trunk_eli.append(trunk_hold[index_-1])

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

		# ----------------------------------------------------------------------------------------------------------------
		# Agglomerative Hierarchical Clustering
		targets_position = np.array([targets[i][0] for i in range(len(targets))])

		watch_1 = col_ind[self.id]
		# print("watch_1: ", watch_1)

		if watch_1 >= 0:

			I_vertex = []
			I_vertex = targets_position[cluster_set[watch_1]]
			# self.Pd = Pd[watch_1]
		else:

			true_watch_1 = int(watch_1) + 100
			I_vertex = np.array([np.array(targets_position[true_watch_1])])

		# print("I_vertex: ", I_vertex)

		# First Construction
		trunk_hold, path = self.Graph_Construction(I_vertex)
		# print("path_origin: ", path)
		# print("trunk_hold_origin: ", trunk_hold)
		
		# Call for Help
		self.Outsourcing(trunk_hold, path)

		# Combine neighbor message
		for neighbor in self.neighbors:

			if neighbor.comc != None:

				# print("neighbor.comc: ", neighbor.comc)
				# print(str(self.id) in neighbor.comc)

				if (str(self.id) in neighbor.comc):

					# print("neighbor.comc: ", neighbor.comc[str(self.id)])

					if len(neighbor.comc[str(self.id)]) != 0:

						for element in neighbor.comc[str(self.id)]:

							if not (np.all((element == path), axis = 1).any()):

								path = np.concatenate((path, neighbor.comc[str(self.id)]), axis = 0)

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
		incircle_A = np.pi*incircle_r**2

		self.incircle = [(incircle_x, incircle_y), incircle_r, incircle_A]

		# Optimal Clustering
		path, trunk = self.Optimal_Cluster(targets, Pd, cluster_set, col_ind, cluster_labels, time_)
		# print("path: " + str(path))
		# print("trunk: " + str(trunk))
		target_points = []

		if trunk[0][0] == -1:

			if (np.shape(path)[0] > 1) and (self.Jx is not None):

				dx = 1.0*\
				np.array([(path[self.Jx][0]-self.virtual_target[0]), (path[self.Jx][1]-self.virtual_target[1])])

			else:
				nearest_index = np.argmin(np.linalg.norm(path[:, np.newaxis] - self.pos, axis=2))
				# print("nearest_index: ", nearest_index)

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
			self.Pd = Pd[self.id]
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
			self.Pd = Pd[self.id]
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
		# print("self.target: " + str(self.target))

		# ANOT
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)

		Observe_list = np.zeros(len(targets))

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])

			if polygon.is_valid and polygon.contains(gemos):

				Observe_list[i] = 1

		Observe_list = np.append(Observe_list, time_)
		self.ANOT = Observe_list[0:len(targets)]

		# ANOT
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/PA/" + self.times
		# filename += self.save_type + str(int(float(self.speed_gain)*100)) + "/PA_" + str(self.id) + ".csv"
		# filename += self.save_type + str(int(float(self.sensing_range))) + "/PA_" + str(self.id) + ".csv"
		
		# KCOV
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/PA/" + self.times
		# filename += "PA_" + str(self.id) + "_" + self.save_type + ".csv"

		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = Observe_list
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def comparsion(self, targets):

		# points = [self.pos]
		points = [self.sweet_spot]

		for neighbor in self.neighbors:

			# points.append(neighbor.pos)
			points.append(neighbor.sweet_spot)

		agents_len = len(points)
		
		for target in targets:

			points.append(target[0])

		points = np.array(points)

		points_len = len(points)

		distances = distance.cdist(points, points)

		# print("distances: " + str(distances))

		# Hungarian Algorithm for 1-1
		cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		cost_matrix = np.array(cost_matrix)
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		watch_1 = col_ind[0]

		self.target = [targets[watch_1]]

	def Hill_Climbing(self, targets, time_):

		env_width, env_height = 25, 25

		initial_box_width = env_width/2
		initial_box_height = env_height/2

		box_width = initial_box_width
		box_height = initial_box_height

		min_box_width = env_width/10
		min_box_height = env_height/10

		# Inscribed circle of FOV
		head_theta = np.arctan(abs(self.perspective[1]/self.perspective[0]))
		incircle_r = (self.R_max*np.sin(self.alpha))/(1 + np.sin(self.alpha))
		incircle_x = self.pos[0] + (self.R_max - incircle_r)*np.sign(self.perspective[0])*np.cos(head_theta)
		incircle_y = self.pos[1] + (self.R_max - incircle_r)*np.sign(self.perspective[1])*np.sin(head_theta)
		incircle_A = np.pi*incircle_r**2

		x, y = incircle_x, incircle_y
		count_pre, H_pre, G_pre = 0, 0, 0

		# Initialization
		for i in range(1):

			count_curr, H_curr, G_curr = 0, 0, 0

			p_pre = self.virtual_target

			for target in targets:

				d = np.linalg.norm(p_pre - target[0])

				if d < incircle_r:

					count_curr += 1

			count_pre = count_curr
			# print("count_pre: " + str(count_pre))

			# points = [self.pos]
			points = [self.sweet_spot]

			for neighbor in self.neighbors:

				# points.append(neighbor.pos)
				points.append(neighbor.sweet_spot)

			agents_len = len(points)
			
			for target in targets:

				points.append(target[0])

			points = np.array(points)

			points_len = len(points)

			distances = distance.cdist(points, points)

			# Hungarian Algorithm for 1-1
			cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
			cost_matrix = np.array(cost_matrix)

			for i in range(np.shape(cost_matrix)[0]):

				for j in range(np.shape(cost_matrix)[1]):

					if cost_matrix[i][j] > 0.5*self.R_max and cost_matrix[i][j] < self.R_max:

						H_curr += cost_matrix[i][j]

			H_pre = H_curr

			observer_no_target, target_not_observed = [], []

			for i in range(np.shape(cost_matrix)[0]):

				if np.all((cost_matrix[i] < incircle_r) == False):

					observer_no_target.append(i)

			for j in range(np.shape(cost_matrix)[1]):

				if np.all((cost_matrix[:,j] < incircle_r) == False):

					target_not_observed.append(j)

			for i in observer_no_target:

				d = np.inf

				for j in target_not_observed:

					if cost_matrix[i,j] < d:

						d = cost_matrix[i,j]

				G_curr += d

			G_pre = G_curr

			# print("cost_matrix: " + str(cost_matrix))
			# print("H_pre: " + str(H_pre))
			# print("G_pre: " + str(G_pre))
			# print(halt)

		# data = np.array([element[0] for element in targets])
		# label_low = np.argmin(np.linalg.norm(data[:, np.newaxis] - np.array([0,0]), axis=2), axis=0)
		# label_high = np.argmax(np.linalg.norm(data[:, np.newaxis] - np.array([0,0]), axis=2), axis=0)

		# x = (data[label_low][0][0] + data[label_high][0][0])/2
		# y = (data[label_low][0][1] + data[label_high][0][1])/2

		# box_width = abs(data[label_low][0][0] - data[label_high][0][0])/2
		# box_width = abs(data[label_low][0][1] - data[label_high][0][1])/2

		# Run
		for i in range(1, 100):

			count_curr, H_curr, G_curr = 0, 0, 0

			# Generate a random position within the box
			x, y = p_pre[0], p_pre[1]
			new_x = np.random.uniform(max(x - box_width, 1), min(x + box_width,25))
			new_y = np.random.uniform(max(y - box_width, 1), min(y + box_width,25))

			# Check if the new position exceeds the environment boundary
			if 1 <= new_x <= env_width and 1 <= new_y <= env_height:

				# Update the position of the point
				x = new_x
				y = new_y

				p_curr = np.array([x, y])

				# Count
				for target in targets:

					d = np.linalg.norm(p_curr - target[0])

					if d < incircle_r:

						count_curr += 1

				self.child = p_curr
				# H
				# points = [self.pos]
				points = [self.child]
				# points = [self.sweet_spot]

				for neighbor in self.neighbors:

					# points.append(neighbor.pos)
					points.append(neighbor.child)
					# points.append(neighbor.sweet_spot)

				if np.any(np.array(points) == None):

					cost_matrix = np.zeros([len(points), len(targets)])
				else:

					agents_len = len(points)
					
					for target in targets:

						points.append(target[0])

					points = np.array(points)

					points_len = len(points)

					distances = distance.cdist(points, points)

					cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
					cost_matrix = np.array(cost_matrix)

				for i in range(np.shape(cost_matrix)[0]):

					for j in range(np.shape(cost_matrix)[1]):

						if cost_matrix[i][j] > 0.5*self.R_max and cost_matrix[i][j] < self.R_max:

							H_curr += cost_matrix[i][j]
				# G
				observer_no_target, target_not_observed = [], []

				for i in range(np.shape(cost_matrix)[0]):

					if np.all((cost_matrix[i] < incircle_r) == False):

						observer_no_target.append(i)

				for j in range(np.shape(cost_matrix)[1]):

					if np.all((cost_matrix[:,j] < incircle_r) == False):

						target_not_observed.append(j)

				for i in observer_no_target:

					d = np.inf

					for j in target_not_observed:

						if cost_matrix[i,j] < d:

							d = cost_matrix[i,j]

					G_curr += d

				# print("observer_no_target: " + str(observer_no_target))
				# print("target_not_observed: " + str(target_not_observed))

				# print("x, y: ", end='')
				# print(x, y)
				# print("count_curr: " + str(count_curr))
				# print("count_pre: " + str(count_pre))
				# print("H_curr: " + str(H_curr))
				# print("H_pre: " + str(H_curr))
				# print("G_curr: " + str(G_curr))
				# print("G_pre: " + str(G_curr))
				# print("\n")

				# Heuristic Step
				if count_curr > count_pre:

					p = p_curr
					p_pre = p_curr
					count_pre = count_curr
					H_pre = H_curr
					G_pre = G_curr
				elif count_curr == count_pre:

					if H_pre > H_curr:

						# print("H_pre: " + str(H_pre))
						# print("H_curr: " + str(H_curr))
						# print(halt)

						p = p_curr
						p_pre = p_curr
						count_pre = count_curr
						H_pre = H_curr
						G_pre = G_curr
					elif H_curr == H_pre:

						if G_pre > G_curr:

							# print("G_pre: " + str(G_pre))
							# print("G_curr: " + str(G_curr))
							# print(halt)

							p = p_curr
							p_pre = p_curr
							count_pre = count_curr
							H_pre = H_curr
							G_pre = G_curr
						else:

							p = p_pre
							# count_pre = count_curr
							# H_pre = H_curr
							# G_pre = G_curr
					else:

						p = p_pre
						# count_pre = count_curr
						# H_pre = H_curr
						# G_pre = G_curr
				else:

					p = p_pre
					# p_pre = p_curr
					# count_pre = count_curr
					# H_pre = H_curr
					# G_pre = G_curr

			# Stop when the box width and height reach the minimum size
			if box_width > min_box_width and box_height > min_box_height:

				# Decrease the box width and height by 1% each iteration, but not below the minimum size
				box_width = max(box_width - initial_box_width/100, min_box_width)
				box_height = max(box_height - initial_box_height/100, min_box_height)
				# box_width = max(box_width*0.99, min_box_width)
				# box_height = max(box_height*0.99, min_box_height)
				# box_width = max(box_width - max(initial_box_width/100, min_box_width), min_box_width)
				# box_height = max(box_height - max(initial_box_height/100, min_box_height), min_box_height)

		self.virtual_target = p
		# print("sweet spot: " + str(self.sweet_spot))
		# print("self.virtual_target: " + str(self.virtual_target))
		# print(halt)
		self.target = [[self.virtual_target, 2, 10]]

	def Kmeans(self, targets, cluster_centers, cluster_labels, time_):

		watch_1 = cluster_labels[self.id]
		self.target = [[cluster_centers[watch_1], 2, 10]]

		# ANOT
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)

		# if (np.isnan(pt[0]).any()):

		# 	pt = self.last_FOV
		# 	polygon = Polygon(pt)
		# else:

		# 	polygon = Polygon(pt)
		# 	self.last_FOV = pt

		Observe_list = np.zeros(len(targets))

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])

			if polygon.is_valid and polygon.contains(gemos):

				Observe_list[i] = 1

		Observe_list = np.append(Observe_list, time_)
		self.ANOT = Observe_list[0:len(targets)]

		# ANOT
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/K/" + self.times
		# filename += self.save_type + str(int(float(self.speed_gain)*100)) + "/K_" + str(self.id) + ".csv"
		# filename += self.save_type + str(int(float(self.sensing_range))) + "/K_" + str(self.id) + ".csv"

		# KCOV
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/K/" + self.times
		# filename += "K_" + str(self.id) + "_" + self.save_type + ".csv"
		
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = Observe_list
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def FuzzyCMeans(self, targets, cluster_centers, cluster_labels, time_):

		watch_1 = cluster_labels[self.id]
		self.target = [[cluster_centers[watch_1], 2, 10]]

		# ANOT
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)

		Observe_list = np.zeros(len(targets))

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])

			if polygon.is_valid and polygon.contains(gemos):

				Observe_list[i] = 1

		Observe_list = np.append(Observe_list, time_)
		self.ANOT = Observe_list[0:len(targets)]

		# ANOT
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/FCM/" + self.times
		# filename += self.save_type + str(int(float(self.speed_gain)*100)) + "/FCM_" + str(self.id) + ".csv"
		# filename += self.save_type + str(int(float(self.sensing_range))) + "/FCM_" + str(self.id) + ".csv"

		# KCOV
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/FCM/" + self.times
		# filename += "FCM_" + str(self.id) + "_" + self.save_type + ".csv"

		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = Observe_list
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def DBSCAN(self, targets, cluster_centers, cluster_labels, time_):

		watch_1 = cluster_labels[self.id]
		self.target = [[cluster_centers[watch_1], 2, 10]]

		# ANOT
		pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
		polygon = Polygon(pt)

		Observe_list = np.zeros(len(targets))

		for (mem, i) in zip(targets, range(len(targets))):

			gemos = Point(mem[0])

			if polygon.is_valid and polygon.contains(gemos):

				Observe_list[i] = 1

		Observe_list = np.append(Observe_list, time_)
		self.ANOT = Observe_list[0:len(targets)]

		# ANOT
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/DBSCAN/" + self.times
		# filename += self.save_type + str(int(float(self.speed_gain)*100)) + "/DBSCAN_" + str(self.id) + ".csv"
		# filename += self.save_type + str(int(float(self.sensing_range))) + "/DBSCAN_" + str(self.id) + ".csv"

		# KCOV
		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/DBSCAN/" + self.times
		# filename += "DBSCAN_" + str(self.id) + "_" + self.save_type + ".csv"

		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = Observe_list
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def Agglomerative_Hierarchical_Clustering(self, targets):

		# Sample data points
		data = np.array([targets[i][0] for i in range(len(targets))])

		# Custom distance threshold for merging clusters
		threshold = 1.7*self.incircle[1]  # Adjust as needed

		# Initialize cluster assignments for each data point
		num_points = len(data)
		cluster_assignments = list(range(num_points))

		# print("num_points: ", num_points)
		# print("cluster_assignments: ", cluster_assignments)

		# Perform Agglomerative Hierarchical Clustering based on custom threshold
		for i in range(num_points):

			for j in range(i + 1, num_points):

				if distance.euclidean(data[i], data[j]) < threshold:

					cluster_assignments[j] = cluster_assignments[i]

		# print("cluster_assignments: ", cluster_assignments)

		# Get unique cluster IDs
		unique_clusters = set(cluster_assignments)
		# print("unique_clusters: ", unique_clusters)

		# Assign cluster IDs to data points
		cluster_mapping = {cluster_id: [] for cluster_id in unique_clusters}
		# print("cluster_mapping: ", cluster_mapping)

		for i, cluster_id in enumerate(cluster_assignments):

			cluster_mapping[cluster_id].append(i)

		# print("cluster_mapping: ", cluster_mapping)

		# Print cluster assignments
		i = 0
		cluster = {}
		for cluster_id, points in cluster_mapping.items():

			cluster[i] = np.array(points)
			i += 1

		print("Cluster: ", cluster)
		
		return cluster

	def JointProbability(self, targets, time_):

		W = [element[0] for element in targets]

		pos = self.pos; perspective = self.perspective; alpha = self.alpha; R = self.R
		lamb = self.lamb; R_ = R**(lamb+1)

		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)

		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		# d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d = d.transpose()[0]

		q_per = self.PerspectiveQuality(d, W, pos, perspective, alpha)

		for (i, element) in zip(range(len(q_per)), q_per):

			if element >= 0.0:

				q_per[i] = element
			else:

				q_per[i] = 0.0

		q_res = self.ResolutionQuality(d, W, pos, perspective, alpha, R, lamb)

		for (i, element) in zip(range(len(q_res)), q_res):

			if element >= 0.0:

				q_res[i] = element
			else:

				q_res[i] = 0.0

		Q = np.multiply(q_per, q_res)
		T_Joint = np.append(Q, time_)

		# print("Perspective: " + str(q_per))
		# print("Resolution: " + str(q_res))
		# print("Quality: " + str(Q))

		# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"

		# if self.cp:

		# 	filename = "/home/leo/mts/src/QBSM/Data/Joint/Comparison/"
		# else:

		# 	filename = "/home/leo/mts/src/QBSM/Data/Joint/Test/"
		# filename += "Joint_" + str(self.id) + ".csv"

		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = T_Joint
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def SEMST(self, targets, sv):

		start_vertex = sv

		targets = [targets[i][0] for i in range(len(targets))]
		# print("targets: " + str(targets) + "\n")

		# Calculate the pairwise distances between targets
		distances = distance.cdist(targets, targets)
		# print("distances: " + str(distances) + "\n")

		# # Create a sparse adjacency matrix from the distances
		# adj_matrix = csr_matrix(distances)

		# # Compute the minimum spanning tree using Kruskal's algorithm
		# mst = minimum_spanning_tree(adj_matrix)

		# # Extract the edges from the MST
		# edges = np.array(mst.nonzero()).T

		# # Define the edges of the minimum spanning tree
		# mst_edges = [tuple(edge) for edge in edges]

		# # Define the weights(distance like EMST) of the edges in the minimum spanning tree
		# mst_weights = [mst.toarray().astype(float)[index[0], index[1]] for index in mst_edges]

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

		# print("MST: " + str(mst_edges) + "\n")
		# print("Weights: " + str(mst_weights) + "\n")

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

		# print("Modified MST: " + str(modified_edges) + "\n")
		# print("Modified Weights: " + str(modified_weights) + "\n")

		return modified_edges, modified_weights

	def Hamiltonian_Path(self, targets, start_vertex):

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

		# modified_edges = np.array(modified_edges)
		
		# if len(modified_edges) == 0:

		# 	Hamiltonian_Path = []

		# 	for i in range(len(points)):

		# 		Hamiltonian_Path.append([i])
		# else:

			# for i in range(len(points)):

			# 	if not np.any(np.logical_or((i == modified_edges)[:,0], (i == modified_edges)[:,1])):

			# 		Hamiltonian_Path.append([i])

			# # print("Hamiltonian_Path: " + str(Hamiltonian_Path))

	def circumcenter(self, targets):

		for i in range(0, len(targets)):

			# globals()["x" + str(i+1)] = targets[i][0][0]
			# globals()["y" + str(i+1)] = targets[i][0][1]
			globals()["x" + str(i+1)] = targets[i][0]
			globals()["y" + str(i+1)] = targets[i][1]

		if (x2 - x1)*(y3 - y1) == (y2 - y1)*(x3 - x1):

			avg_x = (x1 + x2 + x3)/3
			avg_y = (y1 + y2 + y3)/3
			geometric_center = np.array([avg_x, avg_y])
			
			r = 0.0
			points = np.array([(x1, y1), (x2, y2), (x3, y3)])

			for i in range(len(points)):

				dist = np.linalg.norm(geometric_center - points[i])

				if dist >= r:

					r = dist

			center_x = avg_x
			center_y = avg_y
			radius = r
		else:
			
			d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
			center_x = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
			center_y = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

			radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5

		return center_x, center_y, radius

	def calculate_tangent_angle(self, circle_center, circle_radius, point):

		distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)

		if (distance - circle_radius) <= 0.5 or distance <= circle_radius:

			angle = 30*(np.pi/180)
		else:

			adjcent = np.sqrt(distance**2 - circle_radius**2)
			angle = 2*np.arctan(circle_radius/adjcent)

		return angle

	def StageAssignment(self):

		range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		# range_max = self.R*cos(self.alpha)

		if self.centroid is not None:

			range_local_best = (np.linalg.norm(np.asarray(self.centroid) - self.pos))
			r = range_max*range_local_best/(range_max+range_local_best)\
				+ range_local_best*range_max/(range_max+range_local_best)

			if self.stage == 1:

				r = max(r, range_max - sqrt(1/(2*self.target[0][1])))
			else:

				r = self.R*cos(self.alpha)

			tmp = 0
			for i in range(len(self.target)):

				dist = np.linalg.norm(self.pos - np.asarray(self.target[i][0]))
				if dist <= r and -dist <= tmp:
					tmp = -dist
					self.stage = 2

			self.r = r

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

		for neighbor in self.neighbors:

			FoV = neighbor.FoV
			quality_map = ne.evaluate("where((quality_map >= FoV), quality_map, 0)")

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV > 0)))
		self.map_plt = np.array(ne.evaluate("where(quality_map != 0, id_ + 1, 0)"))

		return

	def ComputeCentroidal(self, time_):

		translational_force = np.array([0.,0.])
		rotational_force = np.array([0.,0.]).reshape(2,1)
		zoom_force = 0
		centroid = None

		W = self.W[np.where(self.FoV > 0)]; pos = self.pos; lamb = self.lamb; R = self.R; R_ = R**lamb
		alpha = self.alpha; perspective = self.perspective
		
		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)
		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)

		# d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d[np.where(d == 0)] = 1

		F = multivariate_normal([self.target[0][0][0], self.target[0][0][1]],\
								[[self.target[0][1], 0.0], [0.0, self.target[0][1]]])

		x, y = self.map_size[0]*self.grid_size[0], self.map_size[1]*self.grid_size[1]

		if len(self.voronoi[0]) > 0:

			mu_V = np.empty_like([0.0], dtype = np.float64)
			v_V_t = np.empty_like([0, 0], dtype = np.float64)
			delta_V_t = np.empty_like([0.0], dtype = np.float64)
			x_center = np.empty_like([0.0], dtype = np.float64)
			y_center = np.empty_like([0.0], dtype = np.float64)

			# mu_V = np.sum(np.multiply(\
			# 		np.multiply(np.power(d, self.lamb).transpose()[0], F.pdf(W))/(self.R**self.lamb),\
			# 		self.In_polygon))
			# mu_V = np.sum(\
			# 		np.multiply(np.power(d, self.lamb).transpose()[0], F.pdf(W))/(self.R**self.lamb)
			# 		)

			out = np.empty_like(d); F_ = F.pdf(W)
			ne.evaluate("d**lamb", out = out)
			out = out.transpose()[0]
			ne.evaluate("sum((out*F_)/R_)", out = mu_V)
			mu_V = mu_V[0]

			# temp = np.multiply(np.multiply(np.multiply(\
			# 	np.cos(self.alpha) - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0],\
			# 	d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 	F.pdf(self.W)),\
			# 	self.In_polygon)

			# temp = np.multiply(np.multiply(\
			# 	np.cos(self.alpha) - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0],\
			# 	d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 	F.pdf(W))
			# temp = np.array([temp]).transpose()

			# v_V_t =  np.sum(np.multiply(\
			# 		(np.subtract(W, self.pos)/np.concatenate((d,d), axis = 1)),\
			# 		temp), axis = 0)

			d_ = d.transpose()[0]; temp = np.empty_like(d_); F_ = F.pdf(W);
			ne.evaluate("(cos(alpha) - (lamb/R/(lamb+1))*d_)*(d_**lamb/R_)*F_", out = temp)
			temp = np.array([temp]).transpose()

			d_ = np.concatenate((d,d), axis = 1)
			ne.evaluate("sum(((W - pos)/d_)*temp, axis = 0)", out = v_V_t)

			# delta_V_t = np.sum(np.multiply(np.multiply(np.multiply(np.multiply(\
			# 			(1 - np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective)),\
			# 			(1 - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0])),\
			# 			d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 			F.pdf(W)),\
			# 			self.In_polygon))
			# delta_V_t = np.sum(np.multiply(np.multiply(np.multiply(\
			# 			(1 - np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective)),\
			# 			(1 - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0])),\
			# 			d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
			# 			F.pdf(W)))

			d_ = np.concatenate((d,d), axis = 1); F_ = F.pdf(W); out = np.empty_like(F_)
			ne.evaluate("sum(((W - pos)/d_)*perspective, axis = 1)", out = out)
			d_ = d.transpose()[0];
			ne.evaluate("sum((1 - out)*(1 - (lamb/R/(lamb+1))*d_)*(d_**lamb/R_)*F_, axis = 0)", out = delta_V_t)
			delta_V_t = delta_V_t[0]

			v_V = v_V_t/mu_V
			delta_V = delta_V_t/mu_V
			delta_V = delta_V if delta_V > 0 else 1e-10
			alpha_v = acos(1-sqrt(delta_V))
			alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi

			# x_center = np.sum(np.multiply(np.multiply(\
			# 	W[:,0],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))),\
			# 	self.In_polygon))/mu_V

			# y_center = np.sum(np.multiply(np.multiply(\
			# 	W[:,1],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))),\
			# 	self.In_polygon))/mu_V

			# x_center = np.sum(np.multiply(\
			# 	W[:,0],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))))/mu_V

			# y_center = np.sum(np.multiply(\
			# 	W[:,1],\
			# 	(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(W))/(self.R**self.lamb))))/mu_V

			W_x = W[:,0]; W_y = W[:,1]; d_ = d.transpose()[0]; F_ = F.pdf(W);
			ne.evaluate("sum(W_x*(d_**lamb/R_)*F_)", out = x_center); x_center = x_center[0]/mu_V
			ne.evaluate("sum(W_y*(d_**lamb/R_)*F_)", out = y_center); y_center = y_center[0]/mu_V

			if time_ >= 30.00:

				self.Kv = 10

			centroid = np.array([x_center, y_center])
			translational_force += self.Kp*(np.linalg.norm(centroid - self.pos)\
											- self.R*cos(self.alpha))*self.perspective
			rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
									self.perspective[None,:]))  @  (v_V.reshape(2,1))
			zoom_force += -self.Ka*(self.alpha - alpha_v)

		# self.translational_force = translational_force if self.stage != 2 else 0
		# self.perspective_force = np.asarray([rotational_force[0][0], rotational_force[1][0]])
		# self.zoom_force = zoom_force
		self.centroid = centroid

		return

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
			# print("p_dot: " + str(p_dot))

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
		# H = np.empty_like(zoom_force)
		# for t in targets:

		# 	F = multivariate_normal([t[0][0], t[0][1]],\
		# 						[[0.1, 0.0], [0.0, 0.1]])
		# 	F_ = np.array([F.pdf(W)])
		# 	phi = F_.transpose()

		# 	hold = self.FoV[np.where(self.FoV > 0)]; hold = np.array([hold]).transpose()
		# 	# print("hold: ", np.shape(hold))
		# 	# print("phi: ", np.shape(phi))
		# 	# ne.evaluate("sum( hold*phi )", out = H)
		# 	H += ne.evaluate("sum( hold*phi )")

		# print("H: " + str(H))
		# print("self.H: " + str(self.H))
		# print("rate: " + str((H - self.H)/self.H))

		range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		dist = np.linalg.norm(self.target[0][0]-self.pos)
		dist_s = np.linalg.norm(self.target[0][0]-self.sweet_spot)

		# if (H - self.H)/self.H <= 0.025 and H > 1:
		P_gain = min(np.exp(dist/range_max/2), 3)
		# P1_gain = min(3.0 + 3*np.exp(dist_s-3.0), 3.0)
		# P2_gain = min(3.5*dist_s, 2.5)
		# P_gain = min(0.25*np.exp(dist_s), 2.3)
		if dist <= range_max:

			# self.translational_force = 0.1*np.tanh(p_norm)*p_dot
			self.translational_force = 0.08*p_dot
			# self.translational_force = 1.0*P_gain*p_dot
			self.perspective_force = 25*np.asarray([v_dot[0], v_dot[1]])
			# self.perspective_force = 300*np.asarray([v_dot[0], v_dot[1]])
			self.zoom_force = 0.01*a_dot
			self.stage = 2
			self.r = self.R*cos(self.alpha)
		else:

			# self.translational_force = 3*p_dot
			self.translational_force = P_gain*p_dot
			# self.translational_force = 1.0*P_gain*p_dot
			# self.perspective_force = 25*np.asarray([v_dot[0], v_dot[1]])
			self.perspective_force = 3*np.asarray([v_dot[0], v_dot[1]])
			self.zoom_force = 0.01*a_dot
			self.stage = 1
			self.r = self.R*cos(self.alpha)

		print("self.translational_force: " + str(self.translational_force))

		# self.H = H
		# T_Utility = [self.H, time_]
		# print("Quality: " + str(self.FoV))
		# print("Utility: " + str(self.H))
		# print(halt)

		# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# filename = "/home/leo/mts/src/QBSM/Data/Utility/"
		# filename += "Utility_" + str(self.id) + ".csv"
		# with open(filename, "a", encoding='UTF8', newline='') as f:

		# 	row = T_Utility
		# 	writer = csv.writer(f)
		# 	writer.writerow(row)

	def alpha_complex(self, targets):

		points = [target[0] for target in targets]
		points = np.array(points)
		alpha_complex = AlphaComplex(points = points)
		simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = 2.5)

		# print(*simplex_tree.get_skeleton(1))

		Risky_margin = 3.0
		Risky_force = np.array([0., 0.])
		for simplex in simplex_tree.get_skeleton(1):

			if len(simplex[0]) == 2:

				gain = simplex[1]
				i, j = simplex[0]
				mid = np.array([0.5*(points[i][0] + points[j][0]), 0.5*(points[i][1] + points[j][1])])

				dist = np.linalg.norm(mid - self.pos)

				if dist < Risky_margin:

					# Adjust velocities to avoid collision
					# direction = mid - self.pos
					# Risky_force += +gain*(Risky_margin - dist)*(direction/np.linalg.norm(direction))

					direction = np.array((self.pos - mid)/np.linalg.norm(self.pos - mid))
					heading = np.array((self.pos - self.target[0][0])/np.linalg.norm(self.pos - self.target[0][0]))
					theta = np.arccos(np.dot(direction, heading)); trun = (direction*np.sin(theta))/np.linalg.norm(direction*np.sin(theta))
					Risky_force += +(Risky_margin - dist)*trun

		# if self.cp:

		Risky_force = np.array([0., 0.])

		return Risky_force

	def FormationControl(self, targets):

		# points = []; points.append(self.pos)
		# points = np.concatenate((points, [point[0] for point in targets]))

		# distances = distance.cdist(points, points)[0]
		# distances = np.delete(points, 0)
		# weight = 1 - 0.5*( 1 + np.tanh(2*(distances-5.0)) )

		# Risky_force = self.alpha_complex(targets)
		Risky_force = np.zeros(2)

		# print("weight: " + str(weight))

		# Consider which neighbor is sharing the same target and only use them to obtain formation force
		# enemy_force = np.array([0.,0.])

		# for (i, point) in zip(range(len(distances)), targets):

		# 	enemy_force += weight[i]*1*((self.pos - point[0])/(np.linalg.norm(self.pos - point[0])))

		# enemy_norm = np.linalg.norm(enemy_force)

		# # print("enemy_force: " + str(enemy_force))

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
		# print("self.neighbors: ", self.neighbors)
		# print("one_hop_neighbor: ", self.one_hop_neighbor)

		if len(self.neighbors) > 0:

			neighbors_pos = np.concatenate((neighbors_pos, [neighbor.pos for neighbor in self.neighbors]))
			# neighbors_pos = np.concatenate((neighbors_pos, [neighbor.pos for neighbor in self.neighbors\
			# 											if neighbor.id in self.one_hop_neighbor]))
			# print("neighbors_pos: ", neighbors_pos)

			# Calculate the pairwise distances between targets
			distances = distance.cdist(neighbors_pos, neighbors_pos)[0]
			distances = np.delete(distances, 0)
			weight = 1 - 0.5*( 1 + np.tanh(2*(distances-5)) )

			# Consider which neighbor is sharing the same target and only use them to obtain formation force
			neighbor_force = 0.0

			# for (i, neighbor) in zip(range(len(distances)), self.neighbors):

			# 	neighbor_force += weight[i]*((self.pos - neighbor.pos)/(np.linalg.norm(self.pos - neighbor.pos)))
			for neighbor in self.neighbors:

				if neighbor.id in self.one_hop_neighbor:

					neighbor_force += 1.0*((self.pos - neighbor.pos)/(np.linalg.norm(self.pos - neighbor.pos)))

			if (neighbor_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm
				# print("center_force: ", center_force)

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
				# print("center_force_R_normal: ", center_force_R_normal)

				theta_ = np.arccos(np.dot(center_force_R_normal, neighbor_force_normal))

				neighbor_force_transvers = neighbor_force_norm*np.cos(theta_)*center_force_R_normal
				neighbor_force_transvers_norm = np.linalg.norm(neighbor_force_transvers)
				neighbor_force_transvers_normal = neighbor_force_transvers/neighbor_force_transvers_norm
				# neighbor_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(neighbor_force)*np.sin(theta_)
				# print("neighbor_force_transvers_normal: ", neighbor_force_transvers_normal)
			else:

				neighbor_force_transvers = np.zeros(2)
		else:

			neighbor_force_transvers = np.zeros(2)

		# Herding Force ------------------------------------------------------------------------------------------------------------
		if self.cp == "PA":

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
		else:

			herd_force_transvers = np.zeros(2)

		herd_force_transvers = np.zeros(2)
		# Formation Control-------------------------------------------------------------------------------------------------------
		if self.stage == 2:

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

			center_force_norm = np.linalg.norm(np.asarray(self.target[0][0]) - self.pos)

			repulsive_force = (self.pos - np.asarray(self.target[0][0]))\
				/(np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			repulsive_force_norm = np.linalg.norm(self.pos - np.asarray(self.target[0][0]))

			drive_force = enemy_force_transvers + neighbor_force_transvers + herd_force_transvers
			drive_force_norm = np.linalg.norm(drive_force)

			formation_force = (center_force*(drive_force_norm/(center_force_norm + drive_force_norm))\
							+ drive_force*(center_force_norm/(center_force_norm + drive_force_norm)))

			formation_force += repulsive_force*(self.r - np.linalg.norm(self.pos - np.asarray(self.target[0][0])))

			# target_force = (np.asarray(self.target[0][0]) - self.pos)\
			# 	/(np.linalg.norm(np.asarray(self.target[0][0])- self.pos))

			# target_norm = np.linalg.norm(target_force)

			# center_force = (np.asarray(self.target[0][0]) - self.pos)\
			# 	/(np.linalg.norm(np.asarray(self.target[0][0]) - self.pos))

			# formation_force = (target_force*(neighbor_norm/(target_norm+neighbor_norm))\
			# 				+ neighbor_force*(target_norm/(target_norm+neighbor_norm)))

			# formation_force -= (center_force/np.linalg.norm(center_force))*(self.r - self.linalg.norm\
			# 				(self.pos - np.asarray(self.target[0][0])))

			# self.translational_force += (formation_force + enemy_force + Risky_force + herding_transvers)
			self.translational_force += (formation_force)

			return

		else:

			# formation_force = (neighbor_force + enemy_force + Risky_force + herding_transvers)
			formation_force = (enemy_force_transvers + neighbor_force_transvers + herd_force_transvers)
			self.translational_force += formation_force

		print("self.translational_force: " + str(self.translational_force))

		return

	def CBF_Collision_Herding(self, targets, target_velocity):

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
		# print("self.neighbors: ", self.neighbors)
		# print("one_hop_neighbor: ", self.one_hop_neighbor)

		if len(self.neighbors) > 0:

			neighbors_pos = np.concatenate((neighbors_pos, [neighbor.pos for neighbor in self.neighbors]))
			# neighbors_pos = np.concatenate((neighbors_pos, [neighbor.pos for neighbor in self.neighbors\
			# 											if neighbor.id in self.one_hop_neighbor))
			# print("neighbors_pos: ", neighbors_pos)

			# Calculate the pairwise distances between targets
			distances = distance.cdist(neighbors_pos, neighbors_pos)[0]
			distances = np.delete(distances, 0)
			weight = 1 - 0.5*( 1 + np.tanh(2*(distances-5)) )

			# Consider which neighbor is sharing the same target and only use them to obtain formation force
			neighbor_force = 0.0

			for (i, neighbor) in zip(range(len(distances)), self.neighbors):

				neighbor_force += weight[i]*((self.pos - neighbor.pos)/(np.linalg.norm(self.pos - neighbor.pos)))
			# for neighbor in self.neighbors:

			# 	if neighbor.id in self.one_hop_neighbor:

			# 		neighbor_force += 1.0*((self.pos - neighbor.pos)/(np.linalg.norm(self.pos - neighbor.pos)))

			if (neighbor_force != 0.0).all():

				center_force = (np.asarray(self.target[0][0]) - self.pos)
				center_force_norm = np.linalg.norm(center_force)
				center_force_normal = center_force/center_force_norm
				# print("center_force: ", center_force)

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
				# print("center_force_R_normal: ", center_force_R_normal)

				theta_ = np.arccos(np.dot(center_force_R_normal, neighbor_force_normal))

				neighbor_force_transvers = neighbor_force_norm*np.cos(theta_)*center_force_R_normal
				neighbor_force_transvers_norm = np.linalg.norm(neighbor_force_transvers)
				neighbor_force_transvers_normal = neighbor_force_transvers/neighbor_force_transvers_norm
				# neighbor_transvers = np.dot(R, center_force)\
				# 					*np.linalg.norm(neighbor_force)*np.sin(theta_)
				# print("neighbor_force_transvers_normal: ", neighbor_force_transvers_normal)
			else:

				neighbor_force_transvers = np.zeros(2)
		else:

			neighbor_force_transvers = np.zeros(2)

		# Herding Force ------------------------------------------------------------------------------------------------------------
		if self.cp == "PA":

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

					herd_force_transvers = 2.0*herd_force_norm*np.cos(theta_)*center_force_R_normal
					herd_force_transvers_norm = np.linalg.norm(herd_force_transvers)
					herd_force_transvers_normal = herd_force_transvers/herd_force_transvers_norm

					# herding_transvers = np.dot(R, center_force)\
					# 					*np.linalg.norm(np.array(self.Pd) - self.pos)*np.sin(theta_)
		else:

			herd_force_transvers = np.zeros(2)

		# Formation Control-------------------------------------------------------------------------------------------------------
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

		original_translation_force = self.translational_force

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
				h = 1.1*(dist**2 - dt_min**2) - 2*np.dot(d, target_velocity[i])
				G.append([-2*d[0], -2*d[1]])
				H.append(h)

		for neighbor in self.neighbors:

			dist = np.linalg.norm(self.pos - neighbor.pos)

			if dist < communication_range:

				d = (self.pos - neighbor.pos)
				h = 1.1*(dist**2 - da_min**2) - 2*np.dot(d, neighbor.translational_force)
				G.append([-2*d[0], -2*d[1]])
				H.append(h)

		# Herding
		# if len(self.teammate) == 1:
		if self.cp == "PA":

			theta_max = 25*(np.pi/180)
			V = (self.Pd-self.C)/np.linalg.norm(self.Pd-self.C)
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

				self.translational_force = original_translation_force

	def P(self, d, a, In_polygon, P0_I, R, alpha):

		out = np.empty_like(d)
		ne.evaluate("(d - R*cos(alpha))**2", out = out); out = out.transpose()[0];
		out_ = np.empty_like(d.transpose()[0]);
		ne.evaluate("P0_I*exp(-(out/((2*0.2**2)*(R*cos(alpha)**2))))*exp(-((abs(a)-0)**2)/((2*0.2**2)*(alpha**2)))",\
					out = out_)

		return out_

	def P_(self, d, a, In_polygon, P0_B, R, alpha, R_max):

		out = np.empty_like(d)
		ne.evaluate("(abs(d-0.5*R_max)-0.5*R_max)**2", out = out); out = out.transpose()[0];
		out_1 = np.empty_like(d.transpose()[0]);
		ne.evaluate("P0_B*exp(-(out/((2*0.25**2)*(0.5*R_max**2))))*exp(-((abs(a)-alpha)**2)/((2*0.35**2)*(alpha**2)))",\
					out = out_1)

		out = np.empty_like(d)
		ne.evaluate("(d - 0.5*R)**2", out = out); out = out.transpose()[0];
		out_2 = np.empty_like(d.transpose()[0]);
		ne.evaluate("out_1 + P0_B*exp(-(out/((2*0.3**2)*(0.5*R**2))))*exp(-((abs(a)-0)**2)/((2*0.5**2)*(alpha**2)))",\
					out = out_2)

		return out_2

	def EscapeDensity(self, targets, time_):

		# Environment
		# L = 25
		# Wi = 25
		# x_range = np.arange(0, L, 0.1)
		# y_range = np.arange(0, L, 0.1)

		# L = self.map_size[0]
		# Wi = self.map_size[1]
		# x_range = np.arange(0, L, self.grid_size[0])
		# y_range = np.arange(0, L, self.grid_size[1])
		# X, Y = np.meshgrid(x_range, y_range)

		# W = np.vstack([X.ravel(), Y.ravel()])
		# W = W.transpose()

		W = self.W[np.where(self.FoV > 0)]
		pos = self.pos; perspective = self.perspective; alpha = self.alpha; R = self.R
		lamb = self.lamb;

		# Vertices of Boundary of FoV
		A = np.array([self.pos[0], self.pos[1]])
		B = np.array([self.rtop[0], self.rtop[1]])
		C = np.array([self.ltop[0], self.ltop[1]])
		range_max = (self.lamb + 1)/(self.lamb)*self.R

		# Bivariate Normal Distribution
		F1 = multivariate_normal([targets[0][0][0], targets[0][0][1]],\
								[[targets[0][1], 0.0], [0.0, targets[0][1]]])
		F2 = multivariate_normal([targets[1][0][0], targets[1][0][1]],\
								[[targets[1][1], 0.0], [0.0, targets[1][1]]])
		F3 = multivariate_normal([targets[2][0][0], targets[2][0][1]],\
								[[targets[2][1], 0.0], [0.0, targets[2][1]]])

		# Joint Probability

		# Interior
		# P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.2**2)*(self.R*np.cos(self.alpha)**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2)))), IoO)
		P = lambda d, a, IoO, P0: P0*np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, self.R*np.cos(self.alpha)), 2).transpose()[0], (2*0.2**2)*(self.R*np.cos(self.alpha)**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.2**2)*(self.alpha**2))))

		# Boundary
		# P_ = lambda d, a, IoO, P0: P0*np.multiply(np.add(np.add(\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))),\
		# 			P0*np.multiply(np.multiply(\
		# 			np.exp(-np.divide(np.power(np.subtract(d, 0.5*self.R), 2).transpose()[0], (2*0.3**2)*(0.5*self.R**2))),\
		# 			np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.5**2)*(self.alpha**2)))), IoO)
		# 			), IoO)
		P_ = lambda d, a, IoO, P0: P0*np.add(np.add(\
					np.exp(-np.divide(np.power(np.subtract(abs(d-0.5*range_max), 0.5*range_max), 2).transpose()[0], (2*0.25**2)*(0.5*range_max**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), self.alpha), 2), (2*0.35**2)*(self.alpha**2)))),\
					P0*np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, 0.5*self.R), 2).transpose()[0], (2*0.3**2)*(0.5*self.R**2))),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0), 2), (2*0.5**2)*(self.alpha**2)))))

		# Spatial Sensing Quality
		# Q = lambda W, d, IoO, P0: P0*np.multiply(np.multiply((np.divide(\
		# 			np.dot(np.subtract(W, self.pos),self.perspective), d) - np.cos(self.alpha))/(1 - np.cos(self.alpha)),\
		# 			np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
		# 			(np.power(d, self.lamb)/(self.R**(self.lamb+1))))), IoO)
		Q = lambda W, d, IoO, P0: P0*np.multiply((np.divide(\
					np.dot(np.subtract(W, self.pos),self.perspective), d) - np.cos(self.alpha))/(1 - np.cos(self.alpha)),\
					np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
					(np.power(d, self.lamb)/(self.R**(self.lamb+1)))))

		# Points in FoV
		pt = [A+np.array([0, 0.1]), B+np.array([0.1, -0.1]), C+np.array([-0.1, -0.1]), A+np.array([0, 0.1])]
		polygon = Path(pt)
		In_polygon = polygon.contains_points(self.W) # Boolean

		# Distance and Angle of W with respect to self.pos
		# d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		out = np.empty_like(W)
		ne.evaluate("W - pos", out = out)
		x = out[:,0]; y = out[:,1]
		d = np.empty_like(x)
		ne.evaluate("sqrt(x**2 + y**2)", out = d)
		d = np.array([d]).transpose()

		# a = np.arccos( np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective) )
		# a = np.arccos( np.dot(np.subtract(W, self.pos)/np.concatenate((d,d), axis = 1), self.perspective) )
		d_ = np.concatenate((d,d), axis = 1); a = np.empty_like(d.transpose()[0])
		ne.evaluate("sum(((W - pos)/d_)*perspective, axis = 1)", out = a)
		ne.evaluate("arccos(a)", out = a)

		# Cost Function
		P0_I = 0.9
		# JP_Interior = P(d, a, In_polygon, P0_I)
		# HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Interior))
		JP_Interior = self.P(d, a, In_polygon, P0_I, R, alpha)
		F1_ = F1.pdf(W); F2_ = F2.pdf(W); F3_ = F3.pdf(W)
		HW_Interior_1 = ne.evaluate("sum(F1_*JP_Interior)")
		HW_Interior_2 = ne.evaluate("sum(F2_*JP_Interior)")
		HW_Interior_3 = ne.evaluate("sum(F3_*JP_Interior)")
		HW_Interior = ne.evaluate("HW_Interior_1 + HW_Interior_2 + HW_Interior_3")

		P0_B = 0.9
		# JP_Boundary = P_(d, a, In_polygon, P0_B)
		# HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
		# 			+ np.sum(np.multiply(F3.pdf(W), JP_Boundary))
		JP_Boundary = self.P_(d, a, In_polygon, P0_B, R, alpha, range_max)
		HW_Boundary_1 = ne.evaluate("sum(F1_*JP_Boundary)")
		HW_Boundary_2 = ne.evaluate("sum(F2_*JP_Boundary)")
		HW_Boundary_3 = ne.evaluate("sum(F3_*JP_Boundary)")
		HW_Boundary = ne.evaluate("HW_Boundary_1 + HW_Boundary_2 + HW_Boundary_3")

		# Sensing Quality
		P0_Q = 1.0
		d = d.transpose()[0]
		# SQ = Q(W, d, In_polygon, P0_Q)
		# HW_Sensing = np.sum(np.multiply(F1.pdf(W), SQ))\
		# 			+ np.sum(np.multiply(F2.pdf(W), SQ))\
		# 			+ np.sum(np.multiply(F3.pdf(W), SQ))
		q_per = self.PerspectiveQuality(d, W, pos, perspective, alpha)
		q_res = self.ResolutionQuality(d, W, pos, perspective, alpha, R, lamb)
		SQ = ne.evaluate("P0_Q*q_per*q_res")
		HW_SQ = [ne.evaluate("sum(F1_*SQ)"), ne.evaluate("sum(F2_*SQ)"), ne.evaluate("sum(F3_*SQ)")]

		# self.HW_IT = HW_Interior*0.1**2
		# self.HW_BT = HW_Boundary*0.1**2
		# self.HW_Sensing = [np.sum(np.multiply(F1.pdf(W), SQ)), np.sum(np.multiply(F2.pdf(W), SQ)), np.sum(np.multiply(F3.pdf(W), SQ))]
		# self.In_polygon = In_polygon
		self.HW_IT = ne.evaluate("HW_Interior*0.1**2")
		self.HW_BT = ne.evaluate("HW_Boundary*0.1**2")
		self.HW_Sensing = HW_SQ
		self.In_polygon = In_polygon

		# if self.id == 0:

		# 	print(self.HW_Interior)
		# 	print(self.HW_Boundary, "\n")

	def UpdateOrientation(self):

		dir_ = self.target[0][0] - self.pos

		self.perspective += self.perspective_force*self.step
		self.perspective /= np.linalg.norm(self.perspective)
		print("self.perspective_force: ", self.perspective_force)

		transverse = dir_ - self.perspective
		self.perspective += 0.08*transverse
		self.perspective /= np.linalg.norm(self.perspective)

		return

	def UpdateZoomLevel(self):

		if (self.alpha + self.zoom_force*self.step)  <= 20*(np.pi/180):

			self.alpha += self.zoom_force*self.step

		return

	def UpdatePosition(self):

		self.pos += self.translational_force*self.step
		# print("self.pos: ", self.pos)
		# print("self.last_pos: ", self.last_pos)

		# print(np.all(np.isnan(self.pos)))
		if (np.linalg.norm(self.pos - self.last_pos) >= 10) or np.all(np.isnan(self.pos)):

			# print("self.last_pos: ", self.last_pos)
			# self.pos = np.copy(self.last_pos)
			self.pos = copy.deepcopy(self.last_pos)
		else:

			# print("self.pos: ", self.pos)
			# self.last_pos = np.copy(self.pos)
			self.last_pos = copy.deepcopy(self.pos)

		print("self.pos: ", self.pos)
		print("self.last_pos: ", self.last_pos)

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
		# print("range_max: ", range_max)