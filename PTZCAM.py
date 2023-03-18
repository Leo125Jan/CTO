import warnings
warnings.simplefilter(action = 'ignore', category = RuntimeWarning)

import sys
import csv
import random
import numpy as np
from time import sleep, time
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy import ndimage, sparse
from shapely.geometry import Point
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment
from shapely.plotting import plot_polygon, plot_points

class PTZcon():

	def __init__(self, properties, map_size, grid_size,\
					Kv = 40, Ka = 3, Kp = 3, step = 0.1):

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
		self.perspective = properties['perspective']/self.norm(properties['perspective'])
		self.alpha = properties['AngleofView']/180*np.pi
		self.R = properties['range_limit']
		self.lamb = properties['lambda']
		self.color = properties['color']
		self.R_max = (self.lamb + 1)/(self.lamb)*self.R
		self.r = 0
		self.top = 0
		self.ltop = 0
		self.rtop = 0
		self.centroid = None

		# Tracking Configuration
		self.cluster_count = 0
		self.dist_to_cluster = 0
		self.dist_to_targets = 0
		self.Clsuter_Checklist = None
		self.Cluster_Teammate = np.array([None, None])
		self.dispatch_occpied = False

		# Relative Control Law
		self.translation_force = 0  # dynamics of positional changes
		self.perspective_force = 0  # dynamics of changing perspective direction
		self.stage = 1              # 1: Tracker, 2: Formation Cooperative
		self.target = None
		self.target_assigned = -1
		self.step = step
		self.FoV = np.zeros(np.shape(W)[1])
		self.Kv = Kv                # control gain for perspective control law toward voronoi cell
		self.Ka = Ka                # control gain for zoom level control stems from voronoi cell
		self.Kp = Kp                # control gain for positional change toward voronoi cell 
		self.event = np.zeros((self.size[0], self.size[1]))

	def UpdateState(self, targets, neighbors, time_):

		self.neighbors = neighbors
		self.time = time_

		self.UpdateFoV()
		self.polygon_FOV()
		self.EscapeDensity(targets)
		self.UpdateLocalVoronoi()

		self.Cluster_Formation(targets)
		self.Cluster_Assignment(targets)
		# self.Cluster_Assignment(targets, time_)

		event = np.zeros((self.size[0], self.size[1]))
		self.event = self.event_density(event, self.target, self.grid_size)
		
		self.ComputeCentroidal()
		self.StageAssignment()
		self.FormationControl()
		self.UpdateOrientation()
		self.UpdateZoomLevel()
		self.UpdatePosition()
		
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

	def Cluster_Formation(self, targets):

		checklist = np.zeros((len(targets), len(targets)))
		threshold = 2.3
		self.cluster_count = 0

		for i in range(len(targets)):

			for j in range(len(targets)):

				if j != i:

					p1 = np.array([targets[i][0][0], targets[i][0][1]])
					p2 = np.array([targets[j][0][0], targets[j][0][1]])

					dist = self.norm(p1 - p2)

					if dist <= threshold:

						checklist[i][j] = 1
						self.cluster_count += 1
					else:

						checklist[i][j] = 0

		self.Clsuter_Checklist = checklist

		return

	def Cluster_Assignment(self, targets):

		count = 0
		Cluster = []

		if len(targets) == 3:

			cluster_count_ref = 6
			AtoT = 3

		for i in range(np.shape(self.Clsuter_Checklist)[0]):

			nonindex = np.nonzero(self.Clsuter_Checklist[i][:])[0]

			if i > 0:

				for j in nonindex:

					if j < i and i in np.nonzero(self.Clsuter_Checklist[j][:])[0]:

						if self.id == 0:

							pass
					else:

						c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
						c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

						Cluster.append([(c_x, c_y), 1, 10])
			elif i == 0:

				for j in nonindex:
		        
					c_x = 0.5*(targets[i][0][0] + targets[j][0][0])
					c_y = 0.5*(targets[i][0][1] + targets[j][0][1])

					Cluster.append([(c_x, c_y), 1, 10])

		# Calculate dist between each cluster for Hungarian Algorithm
		dist_to_cluster = np.array([100.0000, 100.0000, 100.0000], dtype = float)

		for (mem, i) in zip(Cluster, range(len(Cluster))):

			p1 = np.array([self.pos[0], self.pos[1]])
			p2 = np.array([mem[0][0], mem[0][1]])

			dist_to_cluster[i] = self.norm(p1 - p2)

		self.dist_to_cluster = dist_to_cluster

		if (self.cluster_count == cluster_count_ref):

			x, y = 0, 0
			cert = 0
			score = -np.inf

			for mem in targets:

				x += mem[0][0]
				y += mem[0][1]

			for mem in Cluster:

				p1 = np.array([mem[0][0], mem[0][1]])
				p2 = np.array([x/len(targets), y/len(targets)])

				dist = self.norm(p1 - p2)

				if dist > score and dist > 0:

					cert = np.exp(-0.5*(dist/1.5))

			self.target = [[(x/AtoT, y/AtoT), cert, 10]]

		# Mode Switch Control
		cluster_quality = 0
		Best_quality_ref = 2200

		for (target, i) in zip(targets, range(0,len(targets))):

			F = multivariate_normal([target[0][0], target[0][1]],\
								[[target[1], 0.0], [0.0, target[0][1]]])

			# event = np.zeros((self.size[0], self.size[1]))
			# event1 = self.event_density(event, [target], self.grid_size)
			# event1 = np.transpose(event1)
			cluster_quality += np.sum(np.multiply(F.pdf(self.W), self.FoV))

		print(cluster_quality)

		# print(cluster_quality)
		# Calculate dist between each target for Hungarian Algorithm
		if cluster_quality >= 10:

			pt = [self.pos, self.ltop, self.top, self.rtop, self.pos]
			polygon = Polygon(pt)
			
			dist_to_targets = np.array([100.00, 100.00, 100.00], dtype = float)

			for (mem, i) in zip(targets, range(len(targets))):

				gemos = Point(mem[0])
				if polygon.is_valid and polygon.contains(gemos):

					p1 = np.array([self.pos[0], self.pos[1]])
					p2 = np.array([mem[0][0], mem[0][1]])

					dist_to_targets[i] = self.norm(p1 - p2)

			self.dist_to_targets = dist_to_targets

		if (cluster_quality >= 2.35*Best_quality_ref):

			x, y = 0, 0

			for target in targets:

				x += target[0][0]
				y += target[0][1]

			self.target = [[(x/AtoT, y/AtoT), 1, 10]]
		elif (cluster_quality < 2.35*Best_quality_ref) and \
				(cluster_quality >= 1.6*Best_quality_ref):

			if (len(Cluster) == AtoT):

				cost_matrix = [self.dist_to_cluster]
				for neighbor in self.neighbors:

					cost_matrix = np.concatenate((cost_matrix, [neighbor.dist_to_cluster]),\
													axis = 0)

				row_ind, col_ind = linear_sum_assignment(cost_matrix)
				self.target = [Cluster[col_ind[0]]]

			if (len(Cluster) == AtoT - 1):

				switch_index = np.argmin(self.dist_to_cluster)
				self.target = [Cluster[switch_index]]
				self.Cluster_Teammate = np.array([2, switch_index])

				cost_matrix = [self.dist_to_cluster]
				Teammate_matrix = [self.Cluster_Teammate]
				count = np.ones(len(Cluster))
				count[self.Cluster_Teammate[1]] = 0

				for neighbor in self.neighbors:

					temp1 = neighbor.dist_to_cluster
					cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

					if ((neighbor.Cluster_Teammate != None).all()) and\
						(neighbor.Cluster_Teammate[0] == 2):

						temp2 = neighbor.Cluster_Teammate
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
						count[neighbor.Cluster_Teammate[1]] = 0
					else:
						temp2 = np.array([None, None])
						Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

				if not (count == 0).all() and (Teammate_matrix != None).all():

					dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
					dispatch_index = np.argmin(dist_untracked)

					if dispatch_index == 0:

						self.target = [Cluster[np.nonzero(count)[0][0]]]
						self.Cluster_Teammate = np.array([2, np.nonzero(count)[0][0]])

			# if (len(Cluster) <= AtoT - 2 and len(Cluster) > 0):

			# 	for i in range(np.shape(self.Clsuter_Checklist)[0]):

			# 		temp = self.Clsuter_Checklist[i,:]

			# 		if (temp == 0).all():

			# 			p1 = np.array([self.pos[0], self.pos[1]])
			# 			p2 = np.array([targets[i][0][0], targets[i][0][1]])
			# 			dist = self.norm(p1 - p2)
			# 			escape_index = i
			# 			Cluster.append(targets[i])

			# 	self.dist_to_cluster[1] = dist
			# 	switch_index = np.argmin(self.dist_to_cluster)
			# 	self.target = [Cluster[switch_index]]
			# 	self.Cluster_Teammate = np.array([1, switch_index])
			# 	cost_matrix = [self.dist_to_cluster]
			# 	print(self.Cluster_Teammate)

			# 	if self.dispatch_occpied:

			# 		self.target = [Cluster[0]]
			# 		self.Cluster_Teammate = np.array([1, 0])
			# 	elif (self.Cluster_Teammate[1] == 1) and not self.dispatch_occpied:

			# 		for neighbor in self.neighbors:

			# 			if (neighbor.Cluster_Teammate[1] == 1):

			# 				temp1 = neighbor.dist_to_cluster
			# 				cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)

			# 				over_tracked = cost_matrix[:,0]
			# 				dispatch_index = np.argmin(over_tracked)

			# 				if dispatch_index == 0:

			# 					self.target = [Cluster[0]]
			# 					self.Cluster_Teammate = np.array([1, 0])
			# 					self.dispatch_occpied = True

		elif (cluster_quality < 1.6*Best_quality_ref) and\
				(self.cluster_count != cluster_count_ref):

			switch_index = np.argmin(self.dist_to_targets)
			self.target = [targets[switch_index]]
			self.Cluster_Teammate = np.array([0, switch_index])

			cost_matrix = [self.dist_to_targets]
			Teammate_matrix = [self.Cluster_Teammate]
			count = np.ones(len(target))
			count[self.Cluster_Teammate[1]] = 0

			for neighbor in self.neighbors:

				if (neighbor.Cluster_Teammate[0] == 0) and\
					(neighbor.Cluster_Teammate[1] == self.Cluster_Teammate[1]):

					temp1 = neighbor.dist_to_targets
					cost_matrix = np.concatenate((cost_matrix, [temp1]), axis = 0)
					# temp2 = neighbor.Cluster_Teammate
					# Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)
					# count[neighbor.Cluster_Teammate[1]] = 0

					over_tracked = cost_matrix[:,self.Cluster_Teammate[1]]
					dispatch_index = np.argmin(over_tracked)

					if dispatch_index == 0:

						self.dist_to_targets[self.Cluster_Teammate[1]] = 100
						switch_index = np.argmin(self.dist_to_targets)
						self.target = [targets[switch_index]]
						self.Cluster_Teammate = np.array([0, switch_index])
			# 	else:
			# 		temp2 = np.array([None, None])
			# 		Teammate_matrix = np.concatenate((Teammate_matrix, [temp2]), axis = 0)

			# if not (count == 0).all() and (Teammate_matrix != None).all():

			# 	dist_untracked = cost_matrix[:,np.nonzero(count)[0]]
			# 	dispatch_index = np.argmin(dist_untracked)

			# 	if dispatch_index == 0:

			# 		self.target = [targets[np.nonzero(count)[0][0]]]

			self.dispatch_occpied == False
	'''
	def Cluster_Assignment(self, targets, time_):

		if time_ <= 1:

			x = targets[0][0][0] + targets[1][0][0] + targets[2][0][0]
			y = targets[0][0][1] + targets[1][0][1] + targets[2][0][1]

			self.target = [[(x/3, y/3), 1, 10]]
	'''
	def StageAssignment(self):

		# range_max = 0.85*(self.lamb + 1)/(self.lamb)*self.R*cos(self.alpha)
		range_max = self.R*cos(self.alpha)

		if self.centroid is not None:

			range_local_best = (self.norm(np.asarray(self.centroid) - self.pos))
			r = range_max*range_local_best/(range_max+range_local_best)\
				+ range_local_best*range_max/(range_max+range_local_best)

			if self.stage == 1:

				r = max(r, range_max - sqrt(1/(2*self.target[0][1])))
			else:

				r = self.R*cos(self.alpha)

			tmp = 0
			for i in range(len(self.target)):

				dist = self.norm(self.pos - np.asarray(self.target[i][0]))
				if dist <= r and -dist <= tmp:
					tmp = -dist
					self.stage = 2

			self.r = r

	def UpdateFoV(self):

		q_per = self.PerspectiveQuality()
		q_res = self.ResolutionQuality()
		Q = np.multiply(q_per, q_res)

		quality_map = np.where((q_per > 0) & (q_res > 0), Q, 0)
		self.FoV = quality_map
		
		# for y_map in range(max(int((self.pos[1] - range_max)/self.grid_size[1]), 0),\
		# 		min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[1])):

		# 	x_map = np.arange(max(int((self.pos[0] - range_max)/self.grid_size[0]), 0),
		# 			min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0]))
		# 	q_per = self.PerspectiveQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
		# 	q_res = self.ResolutionQuality(x_map*self.grid_size[0], y_map*self.grid_size[1])
		# 	quality = np.where((q_per > 0) & (q_res > 0), q_per*q_res, 0)

		# 	if quality_map is None:

		# 		quality_map = quality
		# 	else:

		# 		quality_map = np.vstack((quality_map, quality))

		# self.FoV[max(int((self.pos[1] - range_max)/self.grid_size[1]), 0):\
		# 		min(int((self.pos[1] + range_max)/self.grid_size[1]), self.size[0]), \
		#     	max(int((self.pos[0] - range_max)/self.grid_size[0]), 0):\
		#     	min(int((self.pos[0] + range_max)/self.grid_size[0]), self.size[0])]\
		#     	= quality_map
		return

	def PerspectiveQuality(self):

		d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d = d.transpose()[0]

		return (np.divide(np.dot(np.subtract(self.W, self.pos), self.perspective), d) - np.cos(self.alpha))\
				/(1 - np.cos(self.alpha))

	def ResolutionQuality(self):

		d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d = d.transpose()[0]

		return np.multiply((self.R*np.cos(self.alpha) - self.lamb*(d - self.R*np.cos(self.alpha))),\
							(np.power(d, self.lamb)/(self.R**(self.lamb+1))))

	def EscapeDensity(self, targets):

		# Environment
		L = 25
		Wi = 25
		x_range = np.arange(0, L, 0.1)
		y_range = np.arange(0, L, 0.1)
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		W = W.transpose()

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
		P = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
					np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], range_max**2),\
					np.divide(np.power(np.subtract(abs(a), 0.5*self.alpha), 2), 0.5*self.alpha**2)), IoO)

		P_ = lambda d, a, IoO, P0: P0*np.multiply(np.multiply(\
					np.exp(-np.divide(np.power(np.subtract(d, range_max), 2).transpose()[0], 2*0.8**2)),\
					np.exp(-np.divide(np.power(np.subtract(abs(a), 0.5*self.alpha), 2), 2*0.8**2))), IoO)

		# Points in FoV
		pt = [A+np.array([0, 0.1]), B+np.array([0.1, -0.1]), C+np.array([-0.1, -0.1]), A+np.array([0, 0.1])]
		polygon = Path(pt)
		In_polygon = polygon.contains_points(W) # Boolean

		# Distance and Angle of W with respect to self.pos
		d = np.linalg.norm(np.subtract(W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		a = np.arccos( np.dot(np.divide(np.subtract(W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective) )

		# Cost Function
		P0_I = 0.9
		JP_Interior = P(d, a, In_polygon, P0_I)
		HW_Interior = np.sum(np.multiply(F1.pdf(W), JP_Interior))\
					+ np.sum(np.multiply(F2.pdf(W), JP_Interior))\
					+ np.sum(np.multiply(F3.pdf(W), JP_Interior))

		P0_B = 0.9
		JP_Boundary = P_(d, a, In_polygon, P0_B)
		HW_Boundary = np.sum(np.multiply(F1.pdf(W), JP_Boundary))\
					+ np.sum(np.multiply(F2.pdf(W), JP_Boundary))\
					+ np.sum(np.multiply(F3.pdf(W), JP_Boundary))

		self.HW_Interior = HW_Interior*0.1**2
		self.HW_Boundary = HW_Boundary*0.1**2
		self.In_polygon = In_polygon

		# if self.id == 0:

		# 	print(self.HW_Interior)
		# 	print(self.HW_Boundary, "\n")

	def UpdateLocalVoronoi(self):

		quality_map = self.FoV

		for neighbor in self.neighbors:

			quality_map = np.where((quality_map >= neighbor.FoV), quality_map, 0)

		# self.voronoi = np.array(np.where((quality_map != 0) & (self.FoV != 0)))
		self.voronoi = np.array(np.where((self.FoV > 0)))
		self.map_plt = np.array(np.where(quality_map != 0, self.id + 1, 0))

		return

	def ComputeCentroidal(self):

		translational_force = np.array([0.,0.])
		rotational_force = np.array([0.,0.]).reshape(2,1)
		zoom_force = 0
		centroid = None

		d = np.linalg.norm(np.subtract(self.W, self.pos), axis = 1)
		d = np.array([d]).transpose()
		d[np.where(d == 0)] = 1

		F = multivariate_normal([self.target[0][0][0], self.target[0][0][1]],\
								[[self.target[0][1], 0.0], [0.0, self.target[0][1]]])

		x, y = self.map_size[0]*self.grid_size[0], self.map_size[1]*self.grid_size[1]

		if len(self.voronoi[0]) > 0:

			mu_V = 0
			v_V_t = np.array([0, 0], dtype = np.float64)
			delta_V_t = 0
			x_center = 0
			y_center = 0

			mu_V = np.sum(np.multiply(\
					np.multiply(np.power(d, self.lamb).transpose()[0], F.pdf(self.W))/(self.R**self.lamb),\
					self.In_polygon))

			temp = np.multiply(np.multiply(np.multiply(\
				np.cos(self.alpha) - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0],\
				d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
				F.pdf(self.W)),\
				self.In_polygon)
			temp = np.array([temp]).transpose()

			v_V_t =  np.sum(np.multiply(\
					(np.subtract(self.W, self.pos)/np.concatenate((d,d), axis = 1)),\
					temp), axis = 0)
	

			delta_V_t = np.sum(np.multiply(np.multiply(np.multiply(np.multiply(\
						(1 - np.dot(np.divide(np.subtract(self.W, self.pos),np.concatenate((d,d), axis = 1)), self.perspective)),\
						(1 - (self.lamb/self.R/(self.lamb+1))*d.transpose()[0])),\
						d.transpose()[0]**(self.lamb)/(self.R**self.lamb)),\
						F.pdf(self.W)),\
						self.In_polygon))

			v_V = v_V_t/mu_V
			delta_V = delta_V_t/mu_V
			delta_V = delta_V if delta_V > 0 else 1e-10
			alpha_v = acos(1-sqrt(delta_V))
			alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi

			x_center = np.sum(np.multiply(np.multiply(\
				self.W[:,0],\
				(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(self.W))/(self.R**self.lamb))),\
				self.In_polygon))/mu_V

			y_center = np.sum(np.multiply(np.multiply(\
				self.W[:,1],\
				(np.multiply(d.transpose()[0]**(self.lamb),F.pdf(self.W))/(self.R**self.lamb))),\
				self.In_polygon))/mu_V

			# # Control law for maximizing resolution and perspective quality
			# for i in range(len(self.voronoi[0])):

			# 	x_map = self.voronoi[1][i]
			# 	y_map = self.voronoi[0][i]

			# 	x, y = x_map*self.grid_size[0], y_map*self.grid_size[1]
			# 	x_p = np.array([x,y]) - self.pos
			# 	norm = self.norm(x_p)

			# 	if norm == 0: continue

			# 	mu_V += ((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb)
			# 	v_V_t += ((x_p)/norm)*(cos(self.alpha) - \
			# 		( ( self.lamb*norm )/((self.lamb+1)*self.R)))*\
			# 		( (norm**self.lamb)/(self.R**self.lamb) )*self.event[x_map,y_map]
			# 	dist = (1 - (self.lamb*norm)/((self.lamb+1)*self.R))
			# 	dist = dist if dist >= 0 else 0
			# 	delta_V_t += (1 - (((x_p)@self.perspective.T))/norm)\
			# 					*dist*((norm**self.lamb)/(self.R**self.lamb))\
			# 					*self.event[x_map,y_map]
				# x_center += x*( ((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb) )
			# 	y_center += y*(((norm**self.lamb)*self.event[x_map,y_map] )/(self.R**self.lamb))
		    
			# v_V = v_V_t/mu_V
			# delta_V = delta_V_t/mu_V
			# delta_V = delta_V if delta_V > 0 else 1e-10
			# alpha_v = acos(1-sqrt(delta_V))
			# alpha_v = alpha_v if alpha_v > 5/180*np.pi else 5/180*np.pi
		    
			centroid = np.array([x_center, y_center])
			translational_force += self.Kp*(self.norm(centroid - self.pos)\
											- self.R*cos(self.alpha))*self.perspective
			rotational_force += self.Kv*(np.eye(2) - np.dot(self.perspective[:,None],\
									self.perspective[None,:]))  @  (v_V.reshape(2,1))
			zoom_force += -self.Ka*(self.alpha - alpha_v)

		self.translational_force = translational_force if self.stage != 2 else 0
		self.perspective_force = np.asarray([rotational_force[0][0], rotational_force[1][0]])
		self.zoom_force = zoom_force
		self.centroid = centroid

		return

	def FormationControl(self):

		# Consider which neighbor is sharing the same target and only use them to obtain formation force
		neighbor_force = np.array([0.,0.])

		for neighbor in self.neighbors:
			neighbor_force += (self.pos - neighbor.pos)/(self.norm(self.pos - neighbor.pos))

		neighbor_norm = self.norm(neighbor_force)

		if self.stage == 2:

			target_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(self.norm(np.asarray(self.target[0][0])- self.pos))

			target_norm = self.norm(target_force)

			center_force = (np.asarray(self.target[0][0]) - self.pos)\
				/(self.norm(np.asarray(self.target[0][0]) - self.pos))

			formation_force = (target_force*(neighbor_norm/(target_norm+neighbor_norm))\
							+ neighbor_force*(target_norm/(target_norm+neighbor_norm)))

			formation_force -= (center_force/self.norm(center_force))*(self.r - self.norm\
							(self.pos - np.asarray(self.target[0][0])))

			self.translational_force += formation_force 

			return

		else:

			formation_force = neighbor_force
			self.translational_force += formation_force 

			return

	def UpdateOrientation(self):

		self.perspective += self.perspective_force*self.step
		self.perspective /= self.norm(self.perspective)

		return

	def UpdateZoomLevel(self):

		self.alpha += self.zoom_force*self.step

		return

	def UpdatePosition(self):

		self.pos += self.translational_force*self.step

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