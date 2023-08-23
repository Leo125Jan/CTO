import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import csv
import pygame
import random
import numpy as np
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
					side_center, side_center_radius):

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

	for i in range(0, len(targets)):

		globals()["x" + str(i+1)] = targets[i][0][0]
		globals()["y" + str(i+1)] = targets[i][0][1]

	d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
	center_x = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
	center_y = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

	radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5

	return (center_x, center_y), radius

def sidecenter(targets):

	for i in range(0, len(targets)):

		globals()["x" + str(i+1)] = targets[i][0][0]
		globals()["y" + str(i+1)] = targets[i][0][1]

	side_center_1 = np.array([0.5*(x1 + x2), 0.5*(y1 + y2)]); radius_1 = 0.5*np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
	side_center_2 = np.array([0.5*(x1 + x3), 0.5*(y1 + y3)]); radius_2 = 0.5*np.sqrt( (x1-x3)**2 + (y1-y3)**2 )
	side_center_3 = np.array([0.5*(x2 + x3), 0.5*(y2 + y3)]); radius_3 = 0.5*np.sqrt( (x2-x3)**2 + (y2-y3)**2 )

	side_center = [side_center_1, side_center_2, side_center_3]
	side_center_radius = [radius_1, radius_2, radius_3]

	return side_center, side_center_radius

def One_hop_neighbor(points):

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

	return one_hop_neighbors

def Agglomerative_Hierarchical_Clustering(targets, cameras):

	# Sample data points
	data = np.array([targets[i][0] for i in range(len(targets))])

	# Custom distance threshold for merging clusters
	threshold = 1.7*cameras[0].incircle_r  # Adjust as needed

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

def Hungarian(targets, cameras):

	# Agglomerative Hierarchical Clustering
	targets_position = np.array([targets[i][0] for i in range(len(targets))])
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
	alpha = 1.0
	points = []
	# print("self_pos: ", self.pos)
	# print("self_sweet_spot: ", self.sweet_spot)
	# print("points: ", points)

	for camera in cameras:

		# points.append(neighbor.sweet_spot)
		points.append(alpha*camera.pos + (1-alpha)*camera.sweet_spot)

	agents_len = len(points)

	for target in Pd:
	# for target in cluster_center:

		points.append(target)

	points = np.array(points)

	# print("points: " + str(points))

	points_len = len(points)

	if (points_len - agents_len) > agents_len or (points_len - agents_len) == agents_len:

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
	elif (points_len - agents_len) < agents_len:

		distances = distance.cdist(points, points)
		# print("points: " + str(points) + "\n")
		# print("distances: " + str(distances) + "\n")

		cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
		cost_matrix = np.array(cost_matrix)
		# print("cost_matrix: ", cost_matrix)

		for i in range(len(gain)):

			if gain[i] > 1:

				cost_matrix[:,i] *= 1/gain[i]

		flip_ = np.inf*np.ones(agents_len - (points_len - agents_len))
		hold = []
		# print("flip_: ", flip_)

		for i in range(agents_len):

			if i >= points_len - agents_len:

				hold.append(np.hstack((flip_, cost_matrix[i])))
			else:
				hold.append(np.hstack((cost_matrix[i], flip_)))

		cost_matrix = np.array(hold)
		# print("cost_matrix: ", cost_matrix)

	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# print("col_ind: ", col_ind)
	# print("agents_len: ", agents_len)
	# print("points_len: ", points_len)
	# print("points_len - agents_len: ", points_len - agents_len)

	for i in range(len(col_ind)):

		if col_ind[i] < (points_len - agents_len)-1:

			col_ind[i] = col_ind[i]
		elif col_ind[i] > (points_len - agents_len)-1:

			col_ind[i] = col_ind[i] - (agents_len - (points_len - agents_len))

	# print("col_ind: ", col_ind)

	return Pd, cluster_set, col_ind

def K_means(targets, cameras):

	points = []
	for camera in cameras:

		points.append(camera.sweet_spot)
		# points.append(neighbor.pos)

	agents_len = len(points)
	
	for target in targets:

		points.append(target[0])

	points = np.array(points)

	# print("points: " + str(points))

	points_len = len(points)

	distances = distance.cdist(points, points)
	# print("points: " + str(points) + "\n")
	# print("distances: " + str(distances) + "\n")

	# Hungarian Algorithm
	cost_matrix = [row[agents_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < agents_len]
	cost_matrix = np.array(cost_matrix)
	row_ind, col_ind = linear_sum_assignment(cost_matrix)

	# Step 2 - K-Means Algorithm to update Cluster member and its members
	cluster_centers = np.array([targets[element][0] for element in col_ind])
	# print("cluster_centers: " + str(cluster_centers))
	data = np.array([element[0] for element in targets])
	# print("data: " + str(data))
	alpha = 0.3

	for i in range(100):

		# Step 2: Assignment Step - Assign each data point to the nearest centroid
		cluster_labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2), axis=1)

		# Step 3: Update Step - Recalculate the centroids
		mean = np.array([data[cluster_labels == i].mean(axis=0) for i in range(1)])
		# print("mean: " + str(mean))
		# print("cluster_labels: " + str(cluster_labels))

		if (np.isnan(mean[0]).any()):

			# centroids[0] = self.sweet_spot
			# centroids[0] = np.sum(data, axis=0)/len(data)
			pass
		elif len(mean) > 0:

			# new_centroids = (1 - alpha)*cluster_centers[0] + alpha*mean[0]
			new_centroids = np.array([data[cluster_labels == i].mean(axis = 0) for i in range(len(cluster_centers))])

			# print("new_centroids: " + str(new_centroids))

			# Check for convergence
			if np.allclose(cluster_centers, new_centroids):

				break
			else:

				cluster_centers = new_centroids

	# print("labels: " + str(cluster_labels))
	# print("Centroid: " + str(cluster_centers))

	return cluster_labels

if __name__ == "__main__":

	for i in range(1):

		sleep(1)

		pygame.init()

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		cameras = []
		cameras_pos = []

		sensing_range = "4"

		camera0 = { 'id'            :  0,
					'position'      :  np.array([2.0, 2.0]),
					'perspective'   :  np.array([0.5, 0.5]),
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (200, 0, 0)}
		cameras.append(camera0)

		camera1 = { 'id'            :  1,
					'position'      :  np.array([23.0, 2.0]),
					'perspective'   :  np.array([-0.5, 0.5]),
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (0, 200, 0)}
		cameras.append(camera1)

		camera2 = { 'id'            :  2,
					'position'      :  np.array([2.0, 23.0]),
					'perspective'   :  np.array([0.5, -0.5]),
					'AngleofView'   :  20,
					'range_limit'   :  float(sensing_range),
					'lambda'        :  2,
					'color'         : (0, 0, 200)}
		cameras.append(camera2)

		camera3 = { 'id'            :  3,
					'position'      :  np.array([23.0, 23.0]),
					'perspective'   :  np.array([-0.5, -0.5]),
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
		seed_key = {1: "103", 2: "106", 3: "143", 4: "279", 5: "315", 6: "333", 7: "555", 8: "913", 9: "3821", 10: "5214",
					11: "5232", 12: "5246", 13: "5532"}
		# np.random.seed(seed)
		# num_points = 8
		# x_coordinates = np.random.uniform(2, 23, num_points)
		# y_coordinates = np.random.uniform(2, 23, num_points)

		# targets = []
		# for i in range(num_points):

		# 	targets.append([(x_coordinates[i], y_coordinates[i]), 1, 10])

		targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
					[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10]]

		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10],
		# 			[(3.5, 2.5), 1, 10], [(5.2, 9.0), 1, 10], [(6.5, 2.5), 1, 10], [(2.5, 13), 1, 10],
		# 			[(14.5, 18.0), 1, 10], [(2.5, 19.5), 1, 10], [(16.5, 2.5), 1, 10], [(17.5, 8.5), 1, 10]]

		# Start Simulation
		Done = False
		vis = Visualize(map_size, grid_size)
		last = time()

		save_key = 4
		save_type = {1: "Target_Speed/O/", 2: "Sensing_Range/O/", 3: "Target_Speed/T/", 4: "Sensing_Range/T/"}
		Alg_key = {0: "FCM", 1: "K", 2: "PA"}
		speed_gain = "0.50"

		velocities = (float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)  # Random initial velocities (-0.5 to 0.5)

		# cp = Alg_key[int(i)]
		cp = "PA"
		
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

			one_hop_neighbor = One_hop_neighbor(points)

			Pd, cluster_set, col_ind = None, None, None
			# Pd, cluster_set, col_ind = Hungarian(targets, uav_team.members)
			cluster_labels = None
			cluster_labels = K_means(targets, uav_team.members)
			# print("cluster_set, col_ind: ", cluster_set, col_ind)

			past = time()
			for i in range(len(uav_team.members)):

				neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]
				uav_team.members[i].UpdateState(targets, neighbors, one_hop_neighbor, Pd, cluster_set, col_ind, cluster_labels, 
												np.round(time() - last, 2), cp, speed_gain, save_type[save_key])
			print("Simulation Time: " + str(time() - last))
			print("Calculation Time: " + str(time() - past), "\n")

			circumcenter_center, circumcenter_radius = circumcenter(targets)
			side_center, side_center_radius = sidecenter(targets)

			vis.Visualize2D(uav_team.members, event_plt1, targets, circumcenter_center, circumcenter_radius, \
							side_center, side_center_radius)

			# if np.round(time() - last, 2) > 60.00:
			if run_step >= 1000:

				# sys.exit()
				break

			run_step += 1

		pygame.quit()