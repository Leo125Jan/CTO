import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import csv
import pygame
import random
import numpy as np
import skfuzzy as fuzz
from PTZCAM import PTZcon
from time import sleep, time
from scipy.spatial import distance
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import linear_sum_assignment

initialized = False

class UAVs():

    def __init__(self, map_size, resolution):
        
        self.members = []
        self.map_size = map_size
        self.grid_size = resolution

    def AddMember(self, ptz_info):
        
        ptz = PTZcon(ptz_info, self.map_size, self.grid_size)
        self.members.append(ptz)
        
        return

    # inefficient way, might come up with other data structure to manage the swarm 
    def DeleteMember(self, id): 
        
        for i in range(len(self.members)):
            if self.members.id == id:
                del self.members[i]
                break
        return

class Visualize():

	def __init__(self, map_size, grid_size):

		self.size = (np.array(map_size)/np.array(grid_size)).astype(np.int64)
		self.grid_size = grid_size
		self.window_size = np.array(self.size)*4
		self.display = pygame.display.set_mode(self.window_size)
		self.display.fill((0,0,0))
		self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block

		for x in range(0, self.window_size[0], self.blockSize):

		    for y in range(0, self.window_size[1], self.blockSize):

		        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
		        pygame.draw.rect(self.display, (125,125,125), rect, 1)

		pygame.display.update()
	
	def Visualize2D(self, cameras, event_plt, targets, circumcenter_center, circumcenter_radius, \
					side_center, side_center_radius, run_step):

		map_plt = np.zeros(np.shape(event_plt)[0]) - 1

		for i in range(len(cameras)):

			if i == 1:

				for j in range(np.shape(event_plt)[0]):

					if map_plt[j] == 0 and cameras[i].map_plt[j] > 0:

						cameras[i].map_plt[j] = 1

			map_plt = cameras[i].map_plt + map_plt

		count = 0
		for y in range(0, self.window_size[0], self.blockSize):

			for x in range(0, self.window_size[1], self.blockSize):

				dense = event_plt[count]
				w = 0.6
				id = int(map_plt[count])

				if id == -1:
					gray = (1-w)*125 + w*dense
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, (gray, gray, gray), rect, 0)
				elif id not in range(len(cameras)):

					if id == 3:# N: id-2 -> Green of head of blue
						color = ((1-w)*cameras[id-1].color[0] + w*dense,\
								(1-w)*cameras[id-1].color[1] + w*dense,\
								(1-w)*cameras[id-1].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 4:
						color = ((1-w)*cameras[id-4].color[0] + w*dense,\
								(1-w)*cameras[id-4].color[1] + w*dense,\
								(1-w)*cameras[id-4].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 5:
						color = ((1-w)*cameras[id-5].color[0] + w*dense,\
								(1-w)*cameras[id-5].color[1] + w*dense,\
								(1-w)*cameras[id-5].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
				else:
					color = ((1-w)*cameras[id].color[0] + w*dense,\
							(1-w)*cameras[id].color[1] + w*dense,\
							(1-w)*cameras[id].color[2] + w*dense)
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, color, rect, 0)

				count += 1

		for camera in cameras:

			color = (camera.color[0], camera.color[1], camera.color[2])
			center = camera.pos/self.grid_size*self.blockSize

			R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
			pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
			pygame.draw.circle(self.display, color, camera.pos/self.grid_size*self.blockSize, 10)

		for camera in cameras:

			color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
			pygame.draw.polygon(self.display, color, [camera.pos/self.grid_size*self.blockSize, \
			                                            camera.ltop/self.grid_size*self.blockSize, \
			                                            camera.top/self.grid_size*self.blockSize, \
			                                            camera.rtop/self.grid_size*self.blockSize], 2)

		for target in targets:

			pygame.draw.circle(self.display, (0,0,0), np.asarray(target[0])/self.grid_size\
			                    *self.blockSize, 6)

		targets_position = np.array([target[0] for target in targets])
		GCM = np.mean(targets_position, axis = 0)

		# pygame.draw.circle(self.display, (50,40,30), GCM/self.grid_size*self.blockSize, 10)

		# for (center, r) in zip(side_center, side_center_radius):

		# 	pygame.draw.circle(self.display, (129, 128, 157), center/self.grid_size\
		#                     *self.blockSize, r*(40), 2)

		# for camera in cameras:

		# 	color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
		# 	pygame.draw.circle(self.display, color, camera.incircle[0]/self.grid_size\
		#                     *self.blockSize, camera.incircle[1]*(40), 5)

		# pygame.draw.circle(self.display, (183, 158, 158), circumcenter_center/self.grid_size\
		#                     *self.blockSize, circumcenter_radius*(40), 2)

		for camera in cameras:

			color = (camera.color[0]*0.7, camera.color[1]*0.7, camera.color[2]*0.7)
			pygame.draw.circle(self.display, color, np.asarray(camera.target[0][0])/self.grid_size\
			                    *self.blockSize, 3)

		pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                        map_size[1]/grid_size[1]*self.blockSize), width = 3)

		# Text
		text_font = pygame.font.SysFont("Arial", 30)
		img = text_font.render(str(run_step), True, (255, 255, 255))
		self.display.blit(img, (10, 10))

		text_font = pygame.font.SysFont("Arial", 30)
		for camera in cameras:

			pos = (camera.pos/grid_size*self.blockSize) - 30*camera.perspective

			img = text_font.render(str(camera.id), True, (camera.color[0]*0.7, camera.color[1]*0.7, camera.color[2]*0.7))
			self.display.blit(img, (pos[0], pos[1]))

		text_font = pygame.font.SysFont("Arial", 25)
		for (target, i)in zip(targets, range(len(targets))):

			pos = np.asarray(target[0])
			pos = ((pos - [0.5,0.5])/grid_size*self.blockSize)

			img = text_font.render(str(i), True, (0, 0, 0))
			self.display.blit(img, (pos[0], pos[1]))

		pygame.display.flip()

		# print(halt)
	
def norm(arr):

	sum = 0

	for i in range(len(arr)):

	    sum += arr[i]**2

	return sqrt(sum)

def event_density(event, targets, W):

	for target in targets:

		F = multivariate_normal([target[0][0], target[0][1]],\
						[[target[1], 0.0], [0.0, target[1]]])
		
		event += F.pdf(W)

	return 0 + event

def TargetDynamic(x, y):

	dx = np.random.uniform(-0.5, 0.5, 1)
	dy = np.random.uniform(-0.5, 0.5, 1)

	return (x, y)
	#(np.round(float(np.clip(dx/2 + x, 0, 24)),1), np.round(float(np.clip(dy/2 + y, 0, 24)),1))

def circumcenter(targets):

	if len(targets) >= 3:

		for i in range(0, len(targets)):

			globals()["x" + str(i+1)] = targets[i][0][0]
			globals()["y" + str(i+1)] = targets[i][0][1]

		d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
		center_x = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
		center_y = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

		radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5
	else:

		center_x, center_y = 0.0, 0.0
		radius = 0.0


	return (center_x, center_y), radius

def sidecenter(targets):

	if len(targets) >= 3:

		for i in range(0, len(targets)):

			globals()["x" + str(i+1)] = targets[i][0][0]
			globals()["y" + str(i+1)] = targets[i][0][1]

		side_center_1 = np.array([0.5*(x1 + x2), 0.5*(y1 + y2)]); radius_1 = 0.5*np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
		side_center_2 = np.array([0.5*(x1 + x3), 0.5*(y1 + y3)]); radius_2 = 0.5*np.sqrt( (x1-x3)**2 + (y1-y3)**2 )
		side_center_3 = np.array([0.5*(x2 + x3), 0.5*(y2 + y3)]); radius_3 = 0.5*np.sqrt( (x2-x3)**2 + (y2-y3)**2 )

		side_center = [side_center_1, side_center_2, side_center_3]
		side_center_radius = [radius_1, radius_2, radius_3]
	else:

		side_center = [np.array([0,0]), np.array([0,0]), np.array([0,0])]
		side_center_radius = [0,0,0]

	return side_center, side_center_radius

def One_hop_neighbor(points):

	if len(points) > 2:

		# Generate random points
		points = points

		# Create Delaunay Triangulation
		tri = Delaunay(points)

		# Find one-hop neighbors for each point
		one_hop_neighbors = [[] for _ in range(len(points))]

		# print("one_hop_neighbors: ", one_hop_neighbors)

		for simplex in tri.simplices:

			# print("simplex: ", simplex)

			for point_index in simplex:

				for neighbor_index in simplex:

					if point_index != neighbor_index and neighbor_index not in one_hop_neighbors[point_index]:

						one_hop_neighbors[point_index].append(neighbor_index)

		# print("one_hop_neighbors: ", one_hop_neighbors)
	elif len(points) == 2:

		one_hop_neighbors = [[1], [0]]
	else:

		one_hop_neighbors = [[1], [0]]

	return one_hop_neighbors


def Agglomerative_Partitional_Clustering(targets, cameras):

	# Sample data points
	data = np.array([target[0] for target in targets])

	# Custom distance threshold for merging clusters
	threshold = 1.5*cameras[0].incircle_r  # Adjust as needed

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

	# print("Cluster: ", cluster)
	
	return cluster

def Agglomerative_Hierarchical_Clustering(targets, cameras):

	# Sample data points
	data_points = np.array([target[0] for target in targets])

	# Custom distance threshold for merging clusters
	threshold = 1.5*cameras[0].incircle_r  # Adjust as needed
	# threshold = 3.0*cameras[0].incircle_r  # Adjust as needed
	# print("threshold: ", threshold)

	# Initialize each data point as its own cluster
	clusters = [[point] for point in data_points]
	cluster_mapping = {index_: [index_] for (index_, element) in enumerate(data_points)}
	# cluster_mapping_save = {str(0): [] for (index_, element) in enumerate(data_points)}
	cluster_mapping_save = {0: [0] for (index_, element) in enumerate(data_points)}
	# print("clusters: ", clusters)
	# print("cluster_mapping: ", cluster_mapping)
	# print("cluster_mapping key: ", set(cluster_mapping.keys()))
	# print("cluster_mapping_save: ", cluster_mapping_save)
	# print("cluster_mapping_save key: ", set(cluster_mapping_save.keys()), "\n")

	# Loop until only one cluster remains
	while (set(cluster_mapping.keys()) != set(cluster_mapping_save.keys())):

		cluster_mapping_save = cluster_mapping.copy()
		# print("cluster_mapping: ", cluster_mapping)
		# print("cluster_mapping_save: ", cluster_mapping_save, "\n")

		# Find the two closest clusters
		min_distance = np.inf
		min_i, min_j = -1, -1

		for i in range(len(clusters)):

			if len(clusters[i]) > 1:

				cluster_center_i = np.mean(clusters[i], axis = 0)
			else:

				cluster_center_i = clusters[i][0]

			for j in range(i + 1, len(clusters)):

				if len(clusters[j]) > 1:

					cluster_center_j = np.mean(clusters[j], axis = 0)
				else:
					cluster_center_j = clusters[j][0]

				dist = np.linalg.norm(cluster_center_i - cluster_center_j)
				# print("dist: ", dist)

				if dist < min_distance and dist < threshold:

					min_distance = dist
					min_i, min_j = i, j

		if min_i != -1 and min_j != -1:

			# print("min_i, min_j: ", min_i, min_j)
		
			# Merge the two closest clusters
			clusters[min_i] += clusters[min_j]
			del clusters[min_j]

			cluster_mapping[min_i] = np.append(cluster_mapping[min_i], cluster_mapping[min_j])
			del cluster_mapping[min_j]

			# Print cluster assignments
			i = 0
			cluster = {}
			for cluster_id, points in cluster_mapping.items():

				cluster[i] = np.array(points)
				i += 1

			cluster_mapping = cluster
		else:

	# 		print("min_i, min_j: ", min_i, min_j)
			pass

		# print("clusters: ", clusters)
		# print("cluster_mapping: ", cluster_mapping)
		# print("cluster_mapping key: ", set(cluster_mapping.keys()))
		# print("cluster_mapping_save: ", cluster_mapping_save)
		# print("cluster_mapping_save key: ", set(cluster_mapping_save.keys()), "\n")
		# print(set(cluster_mapping.keys()) == set(cluster_mapping_save.keys()))

	# print("clusters: ", clusters)
	# print("cluster_mapping: ", cluster_mapping)

	return cluster_mapping

def Hungarian(targets, cameras):

	# Agglomerative Hierarchical Clustering
	targets_position = np.array([target[0] for target in targets])
	GCM = np.mean(targets_position, axis = 0)
	# print("targets_position: ", targets_position)
	# print("global_center of mass: ", GCM)

	cluster_set = Agglomerative_Hierarchical_Clustering(targets, cameras)
	# print("cluster_set: ", cluster_set)

	cluster_center = []
	for key, value in cluster_set.items():

		if len(value) > 1:

			# print("targets_position[value]: ", targets_position[value])
			# print(np.mean(targets_position[value], axis = 0))

			cluster_center.append(np.mean(targets_position[value], axis = 0))
		else:

			cluster_center.append(targets_position[value][0])

	cluster_center = np.array(cluster_center)
	# print("cluster_center: ", cluster_center)

	# Herding Algorithm
	Pd = []; ra = 1; gain = []
	for key, value in cluster_set.items():

		df = (cluster_center[key] - GCM)
		Af = GCM + df + (df/np.linalg.norm(df))*ra*np.sqrt(len(value))

		Pd.append(Af)
		gain.append(len(value))

	# print("Pd: ", Pd)
	# print("gain: ", gain)

	# Hungarian Algorithm to get Clster Center
	# points = [self.sweet_spot]
	alpha = 0.0
	points = []
	# print("self_pos: ", self.pos)
	# print("self_sweet_spot: ", self.sweet_spot)
	# print("points: ", points)

	for camera in cameras:

		# points.append(neighbor.sweet_spot)
		points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

	agents_len = len(points)

	# for target in Pd:
	for target in cluster_center:

		points.append(target)

	points = np.array(points)

	# print("points: " + str(points))

	points_len = len(points)
	targets_len = (points_len - agents_len)

	if targets_len > agents_len or targets_len == agents_len:

		distances = distance.cdist(points, points)
		# print("points: " + str(points) + "\n")
		# print("distances: " + str(distances) + "\n")

		# Hungarian Algorithm
		cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		cost_matrix = np.array(cost_matrix)
		# print("cost_matrix: ", cost_matrix)

		for i in range(len(gain)):

			if gain[i] > 1:

				cost_matrix[:,i] *= 1/gain[i]
		# print("cost_matrix: ", cost_matrix)

		row_ind, col_ind = linear_sum_assignment(cost_matrix)
	elif targets_len < agents_len:

		# print("points_len: ", points_len)
		# print("targets_len: ", targets_len)

		distances = distance.cdist(points, points)
		# print("points: " + str(points) + "\n")
		# print("distances: " + str(distances) + "\n")

		cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		cost_matrix = np.array(cost_matrix)
		# print("cost_matrix: ", cost_matrix)

		for i in range(len(gain)):

			if gain[i] > 1:

				cost_matrix[:,i] *= 1/gain[i]

		# flip_ = np.inf*np.ones(agents_len - (points_len - agents_len))
		# hold = []
		# # print("flip_: ", flip_)

		# for i in range(agents_len):

		# 	if i >= points_len - agents_len:

		# 		hold.append(np.hstack((flip_, cost_matrix[i])))
		# 	else:
		# 		hold.append(np.hstack((cost_matrix[i], flip_)))

		# cost_matrix = np.array(hold)
		# print("cost_matrix: ", cost_matrix)

		# hold = np.inf*np.ones([agents_len, agents_len*targets_len])

		# for i in range(agents_len):

		# 	hold[i,i*targets_len:(i+1)*targets_len] = cost_matrix[i,:]

		# print("hold: ", hold)
		# cost_matrix = hold

		# row_ind, col_ind = linear_sum_assignment(cost_matrix)
		# col_ind = col_ind%targets_len

		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		col_sol = {str(i): [] for i in range(agents_len)}

		for (row, col) in zip(row_ind, col_ind):

			col_sol[str(row)] = col

		# List with one missing number
		sequence_num = list(range(0, agents_len))
		missing_numbers = [num for num in sequence_num if num not in row_ind]
		missing_numbers_hold = missing_numbers
		# print("missing_number: ", missing_numbers)

		# Only for excessive assignment
		# points_ = []
		# for i_ in missing_numbers:

		# 	points_.append(cameras[i_].sweet_spot)

		# agents_len_ = len(points_)

		# for position_ in targets_position:

		# 	points_.append(position_)

		# points_ = np.array(points_)
		# points_len_ = len(points_)

		# distances_ = distance.cdist(points_, points_)
		# cost_matrix = [row[agents_len_:points_len_] for (row, i) in zip(distances_, range(len(distances_))) if i < agents_len_]
		# cost_matrix_missing = np.array(cost_matrix)

		# row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
		# # print("row_ind_missing: ", row_ind_missing)
		# # print("col_ind_missing: ", col_ind_missing)

		# for (row, col) in zip(missing_numbers, col_ind_missing):

		# 	col_sol[str(row)] = col-100

		# print("col_sol: ", col_sol)
		# -------------------------------------------------------------------- #

		while len(missing_numbers) != 0:

			cost_matrix_missing = np.array(cost_matrix[missing_numbers])
			# print("cost_matrix_missing: ", cost_matrix_missing)

			row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
			# print("row_ind_missing: ", row_ind_missing)
			# print("col_ind_missing: ", col_ind_missing)

			for (row, col) in zip(missing_numbers, col_ind_missing):

				col_sol[str(row)] = col
				missing_numbers_hold.remove(row)

			print("col_sol: ", col_sol)
			missing_numbers = missing_numbers_hold

		col_ind = np.zeros(agents_len)
		for key_, value_ in col_sol.items():

			col_ind[int(key_)] = value_

		# col_indices = np.hstack((col_indices, col_indices_missing))
		# print("col_ind: ", col_ind)

	# print("col_ind: ", col_ind)
	# print("agents_len: ", agents_len)
	# print("points_len: ", points_len)
	# print("points_len - agents_len: ", points_len - agents_len)

	# for i in range(len(col_ind)):

	# 	if col_ind[i] < (points_len - agents_len)-1:

	# 		col_ind[i] = col_ind[i]
	# 	elif col_ind[i] > (points_len - agents_len)-1:

	# 		col_ind[i] = col_ind[i] - (agents_len - (points_len - agents_len))

	# print("col_ind: ", col_ind)

	return Pd, cluster_set, col_ind

def K_means(targets, cameras):

	points = []
	targets_position = np.array([target[0] for target in targets])

	alpha = 0.0
	for camera in cameras:

		points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

	agents_len = len(points)
	
	for target_position in targets_position:

		points.append(target_position)

	points = np.array(points)

	# print("points: " + str(points))

	points_len = len(points)

	distances = distance.cdist(points, points)
	# print("points: " + str(points) + "\n")
	# print("distances: " + str(distances) + "\n")

	# Hungarian Algorithm
	# cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
	# cost_matrix = np.array(cost_matrix)
	# row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Step 2 - K-Means Algorithm to update Cluster member and its members
	# cluster_centers = np.array([targets_position[element] for element in col_ind])
	# cluster_centers = points[0:4]
	cluster_centers = points[0:agents_len]
	# print("cluster_centers: " + str(cluster_centers))
	data = targets_position
	# print("data: " + str(data))
	alpha = 0.3

	for i in range(100):

		# Step 2: Assignment Step - Assign each data point to the nearest centroid
		cluster_labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2), axis=1)

		# Step 3: Update Step - Recalculate the centroids
		mean = np.array([data[cluster_labels == i].mean(axis=0) for i in range(agents_len)])
		# print("mean: " + str(mean))
		# print("cluster_labels: " + str(cluster_labels))

		for j in range(agents_len):

			count = 0

			if (np.isnan(mean[j]).any()):

				cluster_centers[j] = cameras[j].sweet_spot
			else:

				new_centroids = (1 - alpha)*cluster_centers[j] + alpha*mean[j]
				# new_centroids = np.array([data[cluster_labels == j].mean(axis = 0)])

				# print("new_centroids: " + str(new_centroids))

				# Check for convergence
				if np.allclose(cluster_centers, new_centroids):

					count += 1
				else:

					cluster_centers[j] = new_centroids

			if count == agents_len:

				break

	for (i, item_) in enumerate(cluster_centers):

		if (np.isnan(item_).any()):

			print(True)

			cluster_centers[i] = cameras[i].sweet_spot

	# print("labels: " + str(cluster_labels))
	# print("Centroid: " + str(cluster_centers))

	return cluster_labels

def FuzzyCMeans(targets, cameras):

	points = []
	targets_position = np.array([target[0] for target in targets])

	alpha = 0.0
	for camera in cameras:

		points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

	c = len(points)

	# Fuzzy C-Means algorithm
	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(targets_position.T, c, 2.0, error = 0.001, maxiter = 1000, seed = 42)
	cluster_centers = cntr
	cluster_membership = np.argmax(u, axis = 0)  # Get the cluster membership for each point

	return cluster_centers, cluster_membership


if __name__ == "__main__":

	for i in range(1):

		sleep(1)

		pygame.init()

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		cameras = []
		cameras_pos = []

		sensing_range = "4"

		# Position & Heading Random
		# One
		position_ = np.array([(2.0, 2.0), (23.0, 2.0), (2.0, 23.0), (23.0, 23.0)])
		heading_ = np.array([(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)])

		# Two
		# position_ = np.array([(7.2, 22.8), (12.1, 2.6), (10.3, 21.2), (16.8, 21.4)])
		# heading_ = np.array([(-0.2, 0.8), (0.6, -0.4), (-0.3, 0.5), (0.1, -0.9)])

		# Three
		# position_ = np.array([(4.5, 20.2), (13.7, 2.1), (7.6, 21.3), (21.2, 21.9)])
		# heading_ = np.array([(0.4, -0.7), (-0.8, 0.2), (0.6, -0.1), (-0.3, 0.9)])

		# Four
		# position_ = np.array([(10.8, 3.4), (22.0, 12.9), (5.3, 20.5), (18.1, 5.7)])
		# heading_ = np.array([(0.1, 0.9), (-0.5, -0.6), (0.8, 0.3), (-0.7, 0.4)])

		# Five
		# position_ = np.array([(6.7, 18.8), (17.3, 2.5), (9.2, 19.7), (20.5, 10.2)])
		# heading_ = np.array([(-0.6, -0.2), (0.9, 0.7), (-0.4, 0.8), (0.3, -0.5)])

		# # Empty Cluster
		# position_ = np.array([(2.0, 2.0), (23.0, 7.0), (12.5, 23.0)])
		# heading_ = np.array([(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5)])

		# Empty Observation
		# position_ = np.array([(2.0, 2.0)])
		# heading_ = np.array([(0.5, 0.5)])

		# Excessive Assignment
		# position_ = np.array([(2.0, 2.0), (23.0, 2.0), (12.5, 23.0)])
		# heading_ = np.array([(0.5, 0.5), (-0.5, 0.5), (0.0, -0.5)])

		# CBF Test
		# position_ = np.array([(2.0, 20.0), (23.0, 2.0)])
		# heading_ = np.array([(0.5, 0.5), (-0.5, 0.5)])

		camera0 = { 'id'            :  0,
					'position'      :  position_[0],
					'perspective'   :  heading_[0],
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (200, 0, 0)}
		cameras.append(camera0)

		camera1 = { 'id'            :  1,
					'position'      :  position_[1],
					'perspective'   :  heading_[1],
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (0, 200, 0)}
		cameras.append(camera1)

		camera2 = { 'id'            :  2,
					'position'      :  position_[2],
					'perspective'   :  heading_[2],
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (0, 0, 200)}
		cameras.append(camera2)

		camera3 = { 'id'            :  3,
					'position'      :  position_[3],
					'perspective'   :  heading_[3],
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (255, 150, 0)}
		cameras.append(camera3)
		
		# camera4 = { 'id'            :  4,
		# 			'position'      :  np.array([12.5, 2.0]),
		# 			'perspective'   :  np.array([1.0, 0.0]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (255, 250, 0)}
		# cameras.append(camera4)

		# camera5 = { 'id'            :  5,
		# 			'position'      :  np.array([23.0, 12.5]),
		# 			'perspective'   :  np.array([0.5, 0.5]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (0, 240, 255)}
		# cameras.append(camera5)

		# camera6 = { 'id'            :  6,
		# 			'position'      :  np.array([12.5, 23.0]),
		# 			'perspective'   :  np.array([1.0, 0.0]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (150, 0, 255)}
		# cameras.append(camera6)

		# camera7 = { 'id'            :  7,
		# 			'position'      :  np.array([2.0, 12.5]),
		# 			'perspective'   :  np.array([1.0, 0.0]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (255, 0, 250)}
		# cameras.append(camera7)

		# for i in range(len(cameras)):

		# 	# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# 	filename = "/home/leo/mts/src/QBSM/Data/Utility/"
		# 	# filename += "Data_" + str(i) + ".csv"
		# 	filename += "Utility_" + str(i) + ".csv"

		# 	f = open(filename, "w+")
		# 	f.close()

		# for i in range(len(cameras)):

		# 	# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
		# 	filename = "/home/leo/mts/src/QBSM/Data/Joint/"
		# 	# filename += "Data_" + str(i) + ".csv"
		# 	filename += "Joint_" + str(i) + ".csv"

		# 	f = open(filename, "w+")
		# 	f.close()

		# for i in range(len(cameras)):

		# 	if cp:

		# 		filename = "/home/leo/mts/src/QBSM/Data/Joint/Comparison/"
		# 	else:

		# 		filename = "/home/leo/mts/src/QBSM/Data/Joint/Test/"

		# 	filename += "Joint_" + str(i) + ".csv"

		# 	f = open(filename, "w+")
		# 	f.close()

		# Initialize UAV team with PTZ cameras
		uav_team = UAVs(map_size, grid_size)

		for camera in cameras:
			uav_team.AddMember(camera)
			cameras_pos.append(camera["position"])

		# initialize environment with targets
		size = (map_size/grid_size).astype(np.int64)
		x_range = np.arange(0, map_size[0], grid_size[0])
		y_range = np.arange(0, map_size[1], grid_size[1])
		X, Y = np.meshgrid(x_range, y_range)

		W = np.vstack([X.ravel(), Y.ravel()])
		W = W.transpose()

		# target's [position, certainty, weight, velocity]
		# targets = [[(6.5, 19), 1, 10], [(6.0, 18.0), 1, 10], [(7.0, 18.0), 1, 10]]
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10]]

		# Generate 8 random points
		# One
		# seed_key = {1: "103", 2: "106", 3: "143", 4: "279", 5: "351", 6: "333", 7: "555", 8: "913", 9: "3821", 10: "5214",
		# 			11: "5232", 12: "5246", 13: "5532"}
		# seed_key = {1: "5232", 2: "2545", 3: "143", 4: "279", 5: "351", 6: "333", 7: "555", 8: "913", 9: "3821", 10: "5214",
		# 			11: "5232", 12: "5246", 13: "5532"}

		# Two
		# seed_key = {1: "732", 2: "895", 3: "441", 4: "678", 5: "942", 6: "126", 7: "357", 8: "489", 9: "9021", 10: "6782",
					# 11: "5432", 12: "1246", 13: "9876"}
		seed_key = {1: "732", 2: "985", 3: "441", 4: "678", 5: "942", 6: "126", 7: "357", 8: "489", 9: "9021", 10: "6782",
					11: "5432", 12: "1246", 13: "9876"}

		# Three
		# seed_key = {1: "321", 2: "456", 3: "789", 4: "987", 5: "654", 6: "123", 7: "789", 8: "456", 9: "3210", 10: "9876",
		# 			11: "5432", 12: "2109", 13: "8765"}
		# seed_key = {1: "321", 2: "654", 3: "789", 4: "987", 5: "654", 6: "123", 7: "789", 8: "456", 9: "3210", 10: "9876",
		# 			11: "5432", 12: "2109", 13: "8765"}

		# Four
		# seed_key = {1: "567", 2: "890", 3: "234", 4: "876", 5: "432", 6: "109", 7: "543", 8: "890", 9: "1234", 10: "5678",
		# 			11: "9876", 12: "2109", 13: "6543"}
		# seed_key = {1: "567", 2: "890", 3: "777", 4: "876", 5: "234", 6: "109", 7: "543", 8: "809", 9: "1234", 10: "5678",
		# 			11: "9876", 12: "2109", 13: "6543"}

		# Five
		# seed_key = {1: "432", 2: "765", 3: "890", 4: "123", 5: "456", 6: "789", 7: "234", 8: "567", 9: "9087", 10: "1234",
		# 			11: "5678", 12: "4321", 13: "8765"}
		# seed_key = {1: "414", 2: "595", 3: "890", 4: "123", 5: "546", 6: "765", 7: "234", 8: "567", 9: "9087", 10: "1234",
		# 			11: "5678", 12: "4321", 13: "8765"}

		# Empty Observation
		# seed_key = {1: "321", 2: "456", 3: "789", 4: "987", 5: "654", 6: "123", 7: "789", 8: "456", 9: "3210", 10: "9876",
		# 			11: "5432", 12: "2109", 13: "8765"}

		# np.random.seed(seed)
		# num_points = 8
		# x_coordinates = np.random.uniform(2, 23, num_points)
		# y_coordinates = np.random.uniform(2, 23, num_points)

		# targets = []
		# for i in range(num_points):

		# 	targets.append([(x_coordinates[i], y_coordinates[i]), 1, 10])

		# Normal Usage
		targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
					[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10]]

		# CBF Test
		# targets = [[(9.0, 19.0), 1, 10], [(12.0, 12.0), 1, 10], [(19.0, 9.0), 1, 10]]

		# Empty Cluster
		# targets = [[(7.0, 9.0), 1, 10], [(11.5, 15.0), 1, 10], [(17.0, 9.0), 1, 10]]

		# # Empty Observation
		# targets = [[(12.5, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.5, 13.0), 1, 10]]

		# Excessive Assignment
		# targets = [[(12.5, 12.0), 1, 10], [(13.0, 13.0), 1, 10], [(13.5, 12.0), 1, 10], [(14.5, 10.5), 1, 10],
		# 			[(16.2, 9.5), 1, 10], [(11.5, 10.0), 1, 10], [(11.0, 11.8), 1, 10]]

		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10],
		# 			[(3.5, 2.5), 1, 10], [(5.2, 9.0), 1, 10], [(6.5, 2.5), 1, 10], [(2.5, 13), 1, 10],
		# 			[(14.5, 18.0), 1, 10], [(2.5, 19.5), 1, 10], [(16.5, 2.5), 1, 10], [(17.5, 8.5), 1, 10]]
# 
		# Start Simulation
		Done = False
		vis = Visualize(map_size, grid_size)
		last = time()

		Times = "Five/"
		Alg_key = {0: "FCM", 1: "K", 2: "PA"}
		speed_gain = "0.90"
		save_key = 5
		save_type = {1: "Target_Speed/KC/", 2: "Sensing_Range/KC/", 3: "Target_Speed/APC/", 4: "Sensing_Range/APC/",
						5: speed_gain + "_" + sensing_range}

		velocities = 0.0*(float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)  # Random initial velocities (-0.5 to 0.5)

		# cp = Alg_key[int(i)]
		cp = "FCM"
		
		# for i in range(len(cameras)):

		# 	if cp == "FCM":

		# 		# filename = "/home/leo/mts/src/QBSM/Data/ANOT/FCM/" + save_type[save_key] + speed_gain + "/"
		# 		filename = "/home/leo/mts/src/QBSM/Data/ANOT/FCM/" + save_type[save_key] + sensing_range + "/"
		# 		filename += "FCM_" + str(i) + ".csv"
		# 	elif cp == "K":

		# 		# filename = "/home/leo/mts/src/QBSM/Data/ANOT/K/" + save_type[save_key] + speed_gain + "/"
		# 		filename = "/home/leo/mts/src/QBSM/Data/ANOT/K/" + save_type[save_key] + sensing_range + "/"
		# 		filename += "K_" + str(i) + ".csv"
		# 	elif cp == "PA":

		# 		# filename = "/home/leo/mts/src/QBSM/Data/ANOT/PA/" + save_type[save_key] + speed_gain + "/"
		# 		filename = "/home/leo/mts/src/QBSM/Data/ANOT/PA/" + save_type[save_key] + sensing_range + "/"
		# 		filename += "PA_" + str(i) + ".csv"

		# 	f = open(filename, "w+")
		# 	f.close()

		run_step = 1

		while not Done:

			print("run_step: ", run_step)

			for op in pygame.event.get():

				if op.type == pygame.QUIT:

					Done = True

			# print("run_step: " + str(run_step))

			# if np.round(time() - last, 2) > 30.00 and np.round(time() - last, 2) < 50.00:

			# 	targets[0][0] = (targets[0][0][0] + 0.00, targets[0][0][1] + 0.005)
			# 	targets[1][0] = (targets[1][0][0] - 0.005, targets[1][0][1] - 0.02)
			# 	targets[2][0] = (targets[2][0][0] + 0.03, targets[2][0][1] - 0.04)

			# 	sleep(0.001)
			# elif np.round(time() - last, 2) > 50.00 and np.round(time() - last, 2) < 80.00:

			# 	targets[0][0] = (targets[0][0][0] + 0.03, targets[0][0][1] - 0.01)
			# 	targets[1][0] = (targets[1][0][0] + 0.01, targets[1][0][1] - 0.025)
			# 	targets[2][0] = (targets[2][0][0] + 0.01, targets[2][0][1] - 0.01)

			# 	sleep(0.001)

			# elif np.round(time() - last, 2) > 80.00 and np.round(time() - last, 2) < 110.00:

			# 	targets[0][0] = (targets[0][0][0] + 0.007, targets[0][0][1] - 0.035)
			# 	targets[1][0] = (targets[1][0][0] + 0.0292, targets[1][0][1] - 0.0022)
			# 	targets[2][0] = (targets[2][0][0] + 0.005, targets[2][0][1] - 0.005)

			# 	sleep(0.001)

			# if np.round(time() - last, 2) > 20.00 and np.round(time() - last, 2) < 50.00:
			# if run_step <= 900 and run_step >= 100:
			if run_step >= 100:

				# Simulation parameters
				time_step = 0.1  # Time step in seconds
				min_distance = 0.7  # Minimum distance between points to avoid collision
				boundary_margin = 3  # Minimum distance from the boundary
				tracker_margin = 1  # Minimum distance from the boundary

				# Initialize point positions and velocities
				positions = [target[0] for target in targets]
				# print("positions: " + str(positions))
				# velocities = np.random.rand(4, 2) - 0.5  # Random initial velocities (-0.5 to 0.5)
				# print("velocities: " + str(velocities))

				positions += velocities * time_step
				# print("positions: " + str(positions))

				# Change direction of velocities every 3 seconds
				# if np.round(time()-last, 0)%10 == 0:
				if run_step%100 == 0:
					
					np.random.seed(int(seed_key[int(run_step/100)]))
					velocities = (float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)

					# v_ = np.array([(0,-1), (0,0), (-1,0)])
					# velocities = (float(speed_gain)/0.5)*(v_)

					# v_ = np.array([(1,1), (0,0), (-1,0)])
					# velocities = (float(speed_gain)/0.5)*(v_)

					# print("seed: " + str(int(seed_key[int(run_step/100)])))
					# print("velocities: " + str(velocities))

				# Check for collisions and adjust velocities if necessary
				for i in range(len(positions)):

					# for j in range(i + 1, 4):
					# for j in range(len(positions)):

					# 	if j != i:
						
					# 		dist = np.linalg.norm(positions[i] - positions[j])

					# 		if dist < min_distance:

					# 			# Adjust velocities to avoid collision
					# 			direction = positions[i] - positions[j]
					# 			velocities[i] = +(direction/np.linalg.norm(direction))*0.5
					# 			velocities[j] = -(direction/np.linalg.norm(direction))*0.5

					# for  in range(len(positions)):

					if abs(positions[i, 0] - 0) <= boundary_margin or abs(positions[i, 0] - 25) <= boundary_margin:

						velocities[i, 0] *= -0.5  # Reverse x-direction velocity
					if abs(positions[i, 1] - 0) <= boundary_margin or abs(positions[i, 1] - 25) <= boundary_margin:

						velocities[i, 1] *= -0.5  # Reverse y-direction velocity

				# 	# for k in range(len(positions)):

				# 	# 	dist = np.linalg.norm(positions[i] - cameras_pos[k])

				# 	# 	if dist < tracker_margin:

				# 	# 		# Adjust velocities to avoid collision
				# 	# 		direction = positions[i] - cameras_pos[k]
				# 	# 		velocities[i] = +(direction/np.linalg.norm(direction))*0.8

				for (i, element) in zip(range(len(positions)), positions):

					targets[i][0] = element

			event = np.zeros(np.shape(W)[0])
			event1 = event_density(event, targets, W)
			event_plt1 = ((event - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')

			points = []
			for i in range(len(uav_team.members)):

				points.append(uav_team.members[i].pos)

			if cp == "FCM":

				one_hop_neighbor = None
				Pd, cluster_set, col_ind = None, None, None
				cluster_centers, cluster_labels = FuzzyCMeans(targets, uav_team.members)
			elif cp == "K":

				one_hop_neighbor = None
				Pd, cluster_set, col_ind, cluster_centers = None, None, None, None
				cluster_labels = K_means(targets, uav_team.members)
			elif cp == "PA":

				one_hop_neighbor = One_hop_neighbor(points)

				cluster_labels, cluster_centers = None, None
				Pd, cluster_set, col_ind = Hungarian(targets, uav_team.members)

				# Pd, cluster_set, col_ind, cluster_centers = None, None, None, None
				# cluster_labels = K_means(targets, uav_team.members)

			past = time()
			for i in range(len(uav_team.members)):

				# Decentralized Starting Time
				decentralized_start_time = time()

				neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]
				uav_team.members[i].UpdateState(targets, neighbors, one_hop_neighbor, Pd, cluster_set, col_ind, cluster_labels, 
									cluster_centers, np.round(time() - last, 2), cp, speed_gain, save_type[save_key], Times, velocities)
				decentralized_computing_time = time() - decentralized_start_time
				print("decentralized " + str(i) +" computing_time: ", decentralized_computing_time)
			print("Simulation Time: " + str(time() - last))
			print("Calculation Time: " + str(time() - past), "\n")

			circumcenter_center, circumcenter_radius = circumcenter(targets)
			side_center, side_center_radius = sidecenter(targets)

			vis.Visualize2D(uav_team.members, event_plt1, targets, circumcenter_center, circumcenter_radius, \
							side_center, side_center_radius, run_step)

			# if np.round(time() - last, 2) > 60.00:
			if run_step >= 1000:

				# sys.exit()
				break

			run_step += 1

		pygame.quit()