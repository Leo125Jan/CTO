import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import csv
import copy
import pygame
import random
import numpy as np
import skfuzzy as fuzz
from PTZCAM import PTZcon
from time import sleep, time
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import KMeans, DBSCAN
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import linear_sum_assignment, linprog

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
		self.ANOT = 0
		self.AMTkC = 0

		for x in range(0, self.window_size[0], self.blockSize):

		    for y in range(0, self.window_size[1], self.blockSize):

		        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
		        pygame.draw.rect(self.display, (125,125,125), rect, 1)

		pygame.display.update()
	
	def Visualize2D(self, cameras, event_plt, targets, circumcenter_center, circumcenter_radius, \
					side_center, side_center_radius, run_step, cp, Eps, ANOOT):

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

		# targets_position = np.array([target[0] for target in targets])
		# GCM = np.mean(targets_position, axis = 0)

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
		text_font = pygame.font.SysFont("Arial", 35)
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

		# ANOT
		NOT = []
		for camera in cameras:

			if camera.ANOT is None:

				NOT.append(np.zeros((1,8)))
			else:

				NOT.append(camera.ANOT)

		# Perform logical AND on each column
		# print("NOT: ", NOT)
		column_and = np.logical_or.reduce(NOT, axis=0)
		# print("column_and: ", column_and)

		# Converting boolean array to integer for display
		column_and = column_and.astype(int)
		# print("column_and: ", column_and)
		NOT_ = np.sum(column_and)/8
		# print("NOT: ", NOT)
		self.ANOT += NOT_*8

		text_font = pygame.font.SysFont("Arial", 35)
		img = text_font.render("NOT: " + str(NOT_), True, (255, 255, 255))
		self.display.blit(img, (10, 50))
		# img = text_font.render("ANOT: " + str(np.round(self.ANOT/run_step/8, 3)), True, (255, 255, 255))
		img = text_font.render("ANOT: " + str(ANOOT), True, (255, 255, 255))
		self.display.blit(img, (200, 50))

		# KCOV
		# NOT_ = np.array(NOT)
		# Calculate the number of `1`s in each column
		# count_ones = np.sum(NOT_ == 1, axis=0)
		# print("count_ones: ", count_ones)

		# Calculate the amount of `1`s, `2`s, and the number of elements >= `3`
		# num_ones = np.sum(count_ones >= 1)
		# num_twos = np.sum(count_ones >= 2)
		# num_threes_or_more = np.sum(count_ones >= 3)

		# KCOV = (num_ones+num_twos+num_threes_or_more)
		# print("KCOV: ", KCOV)
		# self.AMTkC += KCOV

		# img = text_font.render("1-cov: " + str(np.round(num_ones/8,3)), True, (255, 255, 255))
		# self.display.blit(img, (10, 50))
		# img = text_font.render("2-cov: " + str(np.round(num_twos/8,3)), True, (255, 255, 255))
		# self.display.blit(img, (230, 50))
		# img = text_font.render("3+-cov: " + str(np.round(num_threes_or_more/8,3)), True, (255, 255, 255))
		# self.display.blit(img, (440, 50))
		# img = text_font.render("AMTkC: " + str(np.round(self.AMTkC/run_step/8/3,3)), True, (255, 255, 255))
		# self.display.blit(img, (260, 10))

		# Name
		if cp == "K":

			img = text_font.render("K-means", True, (255, 255, 255))
			self.display.blit(img, (80, 10))
		elif cp == "PA":

			img = text_font.render("Proposed Method", True, (255, 255, 255))
			self.display.blit(img, (80, 10))
		elif cp == "FCM":

			img = text_font.render("Fuzzy C-means", True, (255, 255, 255))
			self.display.blit(img, (80, 10))
		elif cp == "DBSCAN":

			img = text_font.render("DBSk", True, (255, 255, 255))
			self.display.blit(img, (100, 10))

			img = text_font.render("Eps: "+ str(np.round(Eps*10)), True, (255, 255, 255))
			self.display.blit(img, (210, 10))

		pygame.display.flip()

		# print(halt)

def event_density(event, targets, W):

	for target in targets:

		F = multivariate_normal([target[0][0], target[0][1]],\
						[[target[1], 0.0], [0.0, target[1]]])
		
		event += F.pdf(W)

	return 0 + event

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


def One_hop_neighbor_n(points):

	# Neighbor Computing Time
	neighbor_starting_time = time()

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
	elif len(points) == 2:

		one_hop_neighbors = [[1], [0]]
	else:

		one_hop_neighbors = [[1], [0]]
	# print("one_hop_neighbors: ", one_hop_neighbors)

	neighbor_computing_time = time() - neighbor_starting_time
	# print("neighbor_computing_time: ", neighbor_computing_time)

	return neighbor_computing_time

	# return one_hop_neighbors

'''
def One_hop_neighbor_E(points, cluster_set, col_ind):

	n = len(cluster_set)

	# Find one-hop neighbors for each point
	one_hop_neighbors = [[] for _ in range(len(col_ind))]
	# print("one_hop_neighbors: ", one_hop_neighbors)

	for j in range(n):

		agents = []
		agents_id = []

		for (i, index) in zip(col_ind, range(len(col_ind))):

			if i == j:

				agents.append(points[index])
				agents_id.append(index)

		# print("agents: ", agents)
		# print("agents_id: ", agents_id)

		if len(agents) > 2:

			# Create Delaunay Triangulation
			tri = Delaunay(agents)

			# Find one-hop neighbors for each point
			one_hop_neighbors_temp = [[] for _ in range(len(agents))]

			# print("one_hop_neighbors: ", one_hop_neighbors)

			for simplex in tri.simplices:

				# print("simplex: ", simplex)

				for point_index in simplex:

					for neighbor_index in simplex:

						if point_index != neighbor_index and neighbor_index not in one_hop_neighbors_temp[point_index]:

							one_hop_neighbors_temp[point_index].append(neighbor_index)

			for (index, i) in zip(agents_id, range(len(agents_id))):

				one_hop_neighbors[index] = np.array(agents_id)[one_hop_neighbors_temp[i]]

			# print("one_hop_neighbors: ", one_hop_neighbors)
		elif len(agents) == 2:

				one_hop_neighbors[agents_id[0]].append(np.array(agents_id[1]))
				one_hop_neighbors[agents_id[1]].append(np.array(agents_id[0]))
		else:

			one_hop_neighbors = None

		# print("one_hop_neighbors: ", one_hop_neighbors)
	# print("one_hop_neighbors: ", one_hop_neighbors)

	return one_hop_neighbors
'''

def One_hop_neighbor(points, cluster_set, col_ind, dg_p):

	# Neighbor Computing Time
	neighbor_starting_time = time()

	# Team Formulation -------------------------------
	# Team = [[i] for i in range(len(col_ind))]

	# for (index, value) in enumerate(col_ind):

	# 	for (index_, value_) in enumerate(col_ind):

	# 		if (index_ != index) and (value_ == value):

	# 			Team[index].append(index_)

	# for (index, value) in enumerate(Team):

	# 	Team[index] = np.array(sorted(value))
	# print("Team: ", Team)

	# team_map = {}
	# for index, value in enumerate(col_ind):

	# 	if value not in team_map:

	# 		team_map[value] = [index]
	# 	else:

	# 		team_map[value].append(index)

	# for team_indices in team_map.values():

	# 	sorted_team = np.array(sorted(team_indices))

	# 	for idx in sorted_team:

	# 		Team[idx] = sorted_team
	# print("Team: ", Team)

	# Team = []
	# for i in range(len(col_ind)):

	# 	cluster_id = col_ind[i]
	# 	Team.append(list(np.where(col_ind == cluster_id)[0]))
	# Team = [list(np.where(col_ind == col_ind[i])[0]) for i in range(len(col_ind))]
	# print("Team: ", Team)
	# Team Formulation -------------------------------

	# Find one-hop neighbors for each point --------------------------------------
	# one_hop_neighbors = [[] for _ in range(len(col_ind))]
	# # print("one_hop_neighbors: ", one_hop_neighbors)

	# for (key, value_) in cluster_set.items():

	# 	agents = []
	# 	agents_id = []

	# 	for (index, value) in enumerate(col_ind):

	# 		if value == key:

	# 			agents.append(points[index])
	# 			agents_id.append(index)
	# 	# print("agents: ", agents)
	# 	# print("agents_id: ", agents_id)

	# 	if len(agents) > 2:

	# 		# Create Delaunay Triangulation
	# 		tri = Delaunay(agents)

	# 		# Find one-hop neighbors for each point
	# 		one_hop_neighbors_temp = [[] for _ in range(len(agents))]

	# 		for simplex in tri.simplices:

	# 			# print("simplex: ", simplex)

	# 			for point_index in simplex:

	# 				for neighbor_index in simplex:

	# 					if point_index != neighbor_index and neighbor_index not in one_hop_neighbors_temp[point_index]:

	# 						one_hop_neighbors_temp[point_index].append(neighbor_index)

	# 		for (index, value) in enumerate(agents_id):

	# 			one_hop_neighbors[value] = np.array(agents_id)[one_hop_neighbors_temp[index]].tolist()

	# 		# print("one_hop_neighbors: ", one_hop_neighbors)
	# 	elif len(agents) == 2:

	# 			# print("agents_id[0]: ", agents_id[0])
	# 			# print("agents_id[1]: ", agents_id[1])

	# 			one_hop_neighbors[agents_id[0]].append(agents_id[1])
	# 			one_hop_neighbors[agents_id[1]].append(agents_id[0])

	# 			# print("one_hop_neighbors agents_id[0]: ", one_hop_neighbors[agents_id[0]])
	# 			# print("one_hop_neighbors agents_id[1]: ", one_hop_neighbors[agents_id[1]])
	# 	elif len(agents) == 1:

	# 		if len(points) > 2:

	# 			# Create Delaunay Triangulation
	# 			tri = Delaunay(points)

	# 			for simplex in tri.simplices:

	# 				# print("simplex: ", simplex)

	# 				for point_index in simplex:

	# 					if point_index == agents_id[0]:

	# 						for neighbor_index in simplex:

	# 							if point_index != neighbor_index and neighbor_index not in one_hop_neighbors[point_index]:

	# 								one_hop_neighbors[point_index].append(neighbor_index)
	# 		elif len(points) == 2:

	# 			one_hop_neighbors = [[1], [0]]
	# 		else:

	# 			one_hop_neighbors = [[0]]
	# one_hop_neighbors = [sorted(sublist) for sublist in one_hop_neighbors]
	# # print("one_hop_neighbors: ", one_hop_neighbors)

	Team = [list(np.where(col_ind == col_ind[i])[0]) for i in range(len(col_ind))]
	one_hop_neighbors = copy.deepcopy(Team)
	# print("one_hop_neighbors: ", one_hop_neighbors)
	one_man = np.array([i for i, x in enumerate(one_hop_neighbors) if len(x) == 1])
	# print("one_man: ", one_man)

	# Create Delaunay Triangulation
	tri = Delaunay(points)

	for simplex in tri.simplices:

		# print("simplex: ", simplex)

		for i in range(3):

			for j in range(3):

				if (i != j) and (simplex[i] in one_man) and (simplex[j] not in one_hop_neighbors[simplex[i]]):

					one_hop_neighbors[simplex[i]].append(simplex[j])
	print("Team: ", Team)
	print("one_hop_neighbors: ", one_hop_neighbors)
	# Find one-hop neighbors for each point --------------------------------------

	neighbor_computing_time = time() - neighbor_starting_time
	# print("neighbor_computing_time: ", neighbor_computing_time)


	herding_starting_time_2 = time()
	# --------- Herding Algorithm ----------
	# Find unique elements in the list
	unique_elements = set(col_ind)

	# Create a dictionary to hold the indices of each unique element
	indices_dict = {int(elem): [] for elem in unique_elements}

	# Populate the dictionary with indices
	for index, value in enumerate(col_ind):

		indices_dict[value].append(index)

	# Print the dictionary
	# print("indices_dict: ", indices_dict)
	points = np.array(points)

	Pd = {index: None for index in range(len(points))}
	for key, value in indices_dict.items():

		# dog = []
		# dog.extend(points[value])
		# dog_len = len(dog)
		# print("dog: ", points[value])

		# for dp in dg_p[key]:

		# 	dog.append(dp)

		# print("dog: ", dog)
		# dog = np.array(dog)
		# points_len = len(dog)

		# distances = distance.cdist(dog, dog)
		# cost_matrix = [row[dog_len:points_len] for (row, i) in zip(distances, range(len(distances))) if i < dog_len]
		# cost_matrix = np.array(cost_matrix)
		# print("Herding cost_matrix: \n", cost_matrix)

		# Dog Position
		dog_position = points[value]

		# Calculate the distance matrix
		distance_matrix = np.linalg.norm(dog_position[:, np.newaxis] - dg_p[key], axis=2)
		# print("Herding distance_matrix: \n", distance_matrix)

		row_ind_dp, col_ind_dp = linear_sum_assignment(distance_matrix)
		# print("row_ind_dp: ", row_ind_dp)
		# print("col_ind_dp: ", col_ind_dp)

		for row, col in zip(row_ind_dp, col_ind_dp):

			Pd[value[row]] = dg_p[key][col]

	# print("Pd: ", Pd)
	herding_computing_time_2 = time() - herding_starting_time_2
	print("herding_computing_time_2: ", herding_computing_time_2)

	return Pd, one_hop_neighbors, Team, neighbor_computing_time

def Agglomerative_Hierarchical_Clustering(targets, cameras):

	# Sample data points
	data_points = np.array([target[0] for target in targets])

	# Custom distance threshold for merging clusters
	# threshold = 1.5*cameras[0].incircle_r  # Adjust as needed
	threshold = 2.0*cameras[0].incircle_r  # Adjust as needed
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

	for key in cluster_mapping:

			cluster_mapping[key] = np.array(sorted(cluster_mapping[key]))
	# print("cluster_mapping: ", cluster_mapping)

	return cluster_mapping

def agglomerative_hierarchical_clustering(targets, cameras, type_):

	if type_ == "Single":

		n_clusters = 3

		targets_position = np.array([target[0] for target in targets])

		n_samples = len(targets_position)
		# print("n_samples: ", n_samples)
		clusters = {i: [i] for i in range(n_samples)}
		# print("clusters: ", clusters)
		distances = np.linalg.norm(targets_position[:, None] - targets_position, axis=2)
		# print("distances: \n", distances)
		np.fill_diagonal(distances, np.inf)
		# print("distances: \n", distances)

		while len(clusters) > n_clusters:

			i, j = np.unravel_index(np.argmin(distances), distances.shape)
			# print("clusters: ", clusters)
			clusters[i].extend(clusters[j])
			# print("clusters: ", clusters)
			del clusters[j]
			# print("clusters: ", clusters)

			for k in range(n_samples):

				if k in clusters and k != i:

					distances[i, k] = distances[k, i] = min(distances[i, k], distances[j, k])

				distances[:, j] = distances[j, :] = np.inf

		# print("clusters: ", clusters)

		i = 0
		cluster = {}
		for cluster_id, points in clusters.items():

			cluster[i] = np.array(points)
			i += 1

		cluster_mapping = cluster
		# print("cluster: ", cluster)
		# print("cluster_mapping: ", cluster_mapping)

		for key in cluster_mapping:

			cluster_mapping[key] = np.array(sorted(cluster_mapping[key]))
		# print("cluster_mapping: ", cluster_mapping)
	# elif type_ == "Centroid":

	# 	# Sample data points
	# 	targets_position = np.array([target[0] for target in targets])

	# 	# Custom distance threshold for merging clusters
	# 	# threshold = 8  # Adjust as needed
	# 	threshold = 2.0*cameras[0].incircle_r  # Adjust as needed

	# 	n_samples = len(targets_position)
	# 	# print("n_samples: ", n_samples)
	# 	clusters = [[point] for point in targets_position]
	# 	# print("clusters: \n", clusters)
	# 	cluster_mapping = {i: [i] for i in range(n_samples)}
	# 	cluster_mapping_save = {0: [0] for i in range(n_samples)}
	# 	# print("cluster_mapping: ", cluster_mapping)
	# 	# print("cluster_mapping_save: ", cluster_mapping_save)

	# 	while(set(cluster_mapping.keys()) != set(cluster_mapping_save.keys())):

	# 		cluster_mapping_save = cluster_mapping.copy()

	# 		data_points = np.array([np.mean(cluster, axis = 0) for cluster in clusters])
	# 		distances = np.linalg.norm(data_points[:, None] - data_points, axis=2)
	# 		# print("distances: \n", distances)
	# 		distances[distances >= threshold] = np.inf
	# 		np.fill_diagonal(distances, np.inf)
	# 		# print("distances: \n", distances)

	# 		min_i, min_j = np.unravel_index(np.argmin(distances), distances.shape)
	# 		# print("min_i, min_j: ", min_i, min_j)

	# 		if min_i == 0 and min_j == 0:

	# 			# print("clusters: \n", clusters)
	# 			# print("cluster_mapping: ", cluster_mapping)

	# 			pass
	# 		else:

	# 			clusters[min_i] += clusters[min_j]
	# 			del clusters[min_j]
	# 			# print("clusters: \n", clusters)

	# 			cluster_mapping[min_i] = np.append(cluster_mapping[min_i], cluster_mapping[min_j])
	# 			del cluster_mapping[min_j]
	# 			# print("cluster_mapping: ", cluster_mapping)

	# 			i = 0
	# 			cluster = {}
	# 			for cluster_id, points in cluster_mapping.items():

	# 				cluster[i] = np.array(points)
	# 				i += 1

	# 			cluster_mapping = cluster
	# 			# print("cluster_mapping: ", cluster_mapping)

	# 	for key in cluster_mapping:

	# 			cluster_mapping[key] = np.array(sorted(cluster_mapping[key]))
	# 	# print("cluster_mapping: ", cluster_mapping)
	elif type_ == "Centroid":

		# Sample data points
		targets_position = np.array([target[0] for target in targets])

		# Custom distance threshold for merging clusters
		# threshold = 28  # Adjust as needed
		threshold = 2.0*cameras[0].incircle_r  # Adjust as needed

		n_samples = len(targets_position)
		# print("n_samples: ", n_samples)
		# clusters = [[point] for point in targets_position]
		clusters = {i: [value] for i, value in enumerate(targets_position)}
		# print("clusters: \n", clusters)
		cluster_mapping = {i: [i] for i in range(n_samples)}
		# cluster_mapping_save = {0: [0]}
		# print("cluster_mapping: ", cluster_mapping)
		# print("cluster_mapping_save: ", cluster_mapping_save)

		# data_points = np.array([np.mean(cluster, axis = 0) for cluster in clusters])
		data_points = np.array([np.mean(cluster, axis = 0) for _, cluster in clusters.items()])
		distances = np.linalg.norm(data_points[:, None] - data_points, axis=2)
		# print("distances: \n", distances)
		distances[distances >= threshold] = np.inf
		np.fill_diagonal(distances, np.inf)
		# print("distances: \n", distances, "\n")

		# while(set(cluster_mapping.keys()) != set(cluster_mapping_save.keys())):
		for i in range(n_samples):

			# cluster_mapping_save = cluster_mapping.copy()

			min_i, min_j = np.unravel_index(np.argmin(distances), distances.shape)
			# print("min_i, min_j: ", min_i, min_j)

			if min_i == 0 and min_j == 0:

				# print("clusters: \n", clusters)
				# print("cluster_mapping: ", cluster_mapping)
				break
			else:

				clusters[min_i] += clusters[min_j]
				# clusters[min_i].extend(clusters[min_j])
				del clusters[min_j]
				# print("clusters: ", clusters)

				cluster_mapping[min_i] = np.append(cluster_mapping[min_i], cluster_mapping[min_j])
				del cluster_mapping[min_j]
				# print("cluster_mapping: ", cluster_mapping)

				# Update Distances
				new_centroid = np.mean(clusters[min_i], axis=0)
				for k, value in clusters.items():

					if (k != min_i):

						dist = np.linalg.norm(new_centroid - np.mean(value, axis=0))
						distances[min_i, k] = dist if dist < threshold else np.inf
						distances[k, min_i] = distances[min_i, k]

						# distances[k, min_j] = distances[min_j, k] = np.inf
				# distances[min_i, min_j] = distances[min_j, min_i] = np.inf
				distances[:, min_j] = distances[min_j, :] = np.inf
				# print("distances: \n", distances, "\n")

				# Update Distances
				# new_centroid = np.mean(clusters[min_i], axis=0)
				# for k in range(len(clusters)):

				# 	if (k != min_i) and (k != min_j):

				# 		dist = np.linalg.norm(new_centroid - np.mean(clusters[k], axis=0))
				# 		distances[min_i, k] = dist if dist < threshold else np.inf
				# 		distances[k, min_i] = distances[min_i, k]

				# 		distances[k, min_j] = distances[min_j, k] = np.inf

				# distances[min_i, min_j] = distances[min_j, min_i] = np.inf
				# print("distances: \n", distances, "\n")

		cluster_mapping = {i: np.array(sorted(points)) for i, points in enumerate(cluster_mapping.values())}
		# print("cluster_mapping: ", cluster_mapping)

	return cluster_mapping

def Hungarian(Nc, cluster_center, cluster_set, Re, agents_position):

	agents_len = len(agents_position)
	targets_len = len(cluster_center)
	# print("agents_len: ", agents_len)
	# print("targets_len: ", targets_len)

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
	# print("distance_matrix: \n", distance_matrix)

	row_ind, col_ind = None, None
	if targets_len > agents_len or targets_len == agents_len:

		# print("Re: ", Re)
		# Define the vector
		vector = np.array(Re)

		# Compute the reciprocal of the vector
		reciprocal_vector = 1 / vector

		# Multiply each row of the matrix by the reciprocal vector using broadcasting
		cost_matrix = distance_matrix * reciprocal_vector
		# print("cost_matrix: \n", cost_matrix)
		
		row_ind, col_ind = linear_sum_assignment(cost_matrix)
		# print("col_ind: ", col_ind)
	elif targets_len < agents_len:

		# print("Re: ", Re)
		# Define the vector
		vector = np.array(Re)

		# Compute the reciprocal of the vector
		reciprocal_vector = 1 / vector

		# Multiply each row of the matrix by the reciprocal vector using broadcasting
		cost_matrix = distance_matrix * reciprocal_vector
		# print("cost_matrix: \n", cost_matrix)

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

			# print("col_sol: ", col_sol)
			missing_numbers = missing_numbers_hold
			
		col_ind = np.zeros(agents_len)
		for key_, value_ in col_sol.items():

			col_ind[int(key_)] = value_

	print("col_ind: ", col_ind)

	return col_ind

def BILP(Nc, N, M, cluster_center, cluster_set, Re, agents_position):

	A, B = None, None

	if (N == M):

		# Calculate the distance matrix
		distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
		# print("distance matrix: \n", distance_matrix)

		row, col = np.shape(distance_matrix)
		# C = distance_matrix.reshape((1, row*col))[0]
		# print("C: ", C)
		C = distance_matrix.flatten()
		# print("C: ", C)

		# A Matrix Formulation
		# A = np.zeros(row*col)
		# for i in range(row+col):

		# 	if i == 0:

		# 		for j in range(col):

		# 			A[j] = -1

		# 	if i < row and i > 0:

		# 		temp = np.zeros(row*col)

		# 		for j in range(col):

		# 			temp[i*col+j] = -1
				
		# 		A = np.vstack((A, temp))

		# 	if i >= row:

		# 		temp = np.zeros(row*col)
		# 		temp[i-row:row*col:col] = 1
				
		# 		A = np.vstack((A, temp))

		# print("A: \n", A)

		# A Matrix Formulation
		A = np.zeros((row + col, row * col))

		for i in range(row):

			A[i, i * col:(i + 1) * col] = -1
		for j in range(col):

			A[row + j, j::col] = 1
		# print("A: \n", A)

		# B Formulaton
		# B = np.zeros(row+col)
		# for i in range(row+col):

		# 	if i < row:

		# 		B[i] = -1
		# 	elif i >= row:

		# 		B[i] = Re[i-row
		# print("B: \n", B

		# B Formulation
		B = np.zeros(row + col)
		B[:row] = -1
		B[row:] = Re
		# print("B: \n", B)
	elif (N < M):

		# Calculate the distance matrix
		distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
		# print("distance matrix: \n", distance_matrix)

		# print("Re: ", Re)
		# Define the vector
		vector = np.array(Re)

		# Compute the reciprocal of the vector
		reciprocal_vector = 1 / vector

		# Multiply each row of the matrix by the reciprocal vector using broadcasting
		cost_matrix = distance_matrix * reciprocal_vector
		# print("cost_matrix: \n", cost_matrix)

		row, col = np.shape(cost_matrix)
		C = cost_matrix.reshape((1, row*col))[0]
		# print("C: ", C)

		# A Matrix Formulation
		A = np.zeros(row*col)
		for i in range(row+col):

			if i == 0:

				for j in range(col):

					A[j] = -1

			if i < row and i > 0:

				temp = np.zeros(row*col)

				for j in range(col):

					temp[i*col+j] = -1
				
				A = np.vstack((A, temp))

			if i >= row:

				temp = np.zeros(row*col)
				temp[i-row:row*col:col] = -1
				
				A = np.vstack((A, temp))

		# print("A: ", A)

		# B Formulaton
		B = np.zeros(row+col)
		for i in range(row+col):

			if i < row:

				B[i] = -1
			elif i >= row:

				B[i] = -1

		# print("B: ", B)

	# Bounds for each variable (x0 and x1) as binary (0 or 1)
	bounds = [(0, 1)] * len(C)
	bounds = tuple(bounds)
	# print("bounds: ", bounds)

	# Solve the binary linear programming problem
	res = linprog(C, A_ub=A, b_ub=B, bounds=bounds, method='highs')

	# print("Optimal solution:")
	# print(f"x: {res.x}")
	# print("Optimal value:", res.fun)

	# col_ind
	# col_ind = np.zeros(row)
	# for i in range(row):

	# 	for j in range(col):

	# 		if res.x[i*col+j] > 0:
				
	# 			col_ind[i] = j
	# print("col_ind: ", col_ind)

	x_reshaped = res.x.reshape(row, col)
	col_ind = np.argmax(x_reshaped, axis=1)
	# print("col_ind: ", col_ind)

	# # Count occurrences of each element in the array
	# count = Counter(col_ind)

	# # Number of unique elements
	# num_unique_elements = len(count)

	# # Total length of the array
	# total_length = len(col_ind)

	# # Calculate the number of identical elements
	# num_identical_elements = total_length - num_unique_elements

	# if Nc != num_unique_elements:

	# 	keys_with_large_values = {key: len(value) for key, value in cluster_set.items() if len(value) > 1}
	# 	print("keys_with_large_values: ", keys_with_large_values)
	
	return col_ind

def Allocation(targets, cameras):

	# Agglomerative Hierarchical Clustering
	targets_position = np.array([target[0] for target in targets])
	# print("targets_position: ", targets_position)

	GCM = np.mean(targets_position, axis = 0)
	# print("GCM: ", GCM)

	# Cluster Computing Time
	CT = []
	cluster_starting_time = time()

	# cluster_set = Agglomerative_Hierarchical_Clustering(targets, cameras)
	# print("cluster_set 1: ", cluster_set)
	cluster_set = agglomerative_hierarchical_clustering(targets, cameras, "Centroid")
	print("cluster_set 2: ", cluster_set)

	cluster_computing_time = time() - cluster_starting_time; CT.append(cluster_computing_time)


	cluster_center, Re = [], []
	for key, value in cluster_set.items():

		if len(value) > 1:

			# print("targets_position[value]: ", targets_position[value])
			# print(np.mean(targets_position[value], axis = 0))

			cluster_center.append(np.mean(targets_position[value], axis = 0))
		else:

			cluster_center.append(targets_position[value][0])

		Re.append(len(value))
	# print("Re: ", Re)

	cluster_center = np.array(cluster_center)
	print("cluster_center: ", cluster_center)

	N = len(cameras)
	M = len(targets)
	Nc = len(cluster_set)
	# print("Number of agent: ", N)
	# print("Number of target: ", M)
	# print("Number of cluster: ", Nc)

	alpha = 0.0
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])
	col_ind = None

	# Allocation Computing Time
	allocation_starting_time = time()

	if (N == M):

		col_ind = BILP(Nc, N, M, cluster_center, cluster_set, Re, agents_position)
	elif (N < M) and (N < Nc):

		col_ind = Hungarian(Nc, cluster_center, cluster_set, Re, agents_position)
	elif (N < M) and (N == Nc):

		col_ind = Hungarian(Nc, cluster_center, cluster_set, Re, agents_position)
	elif (N < M) and (N > Nc):

		col_ind = BILP(Nc, N, M, cluster_center, cluster_set, Re, agents_position)
	print("col_ind: ", col_ind)

	allocation_computing_time = time() - allocation_starting_time; CT.append(allocation_computing_time)


	harding_starting_time_1 = time()
	# ---------- Herding Algorithm ----------
	Pd = []; ra = 1;
	dg_p = {key: [] for key in range(len(cluster_set.keys()))}

	if len(cluster_set.keys()) == 1:

		angle = (2*np.pi)/len(cameras)

		for i in range(len(cameras)):

			# Calculate the angle for this vertex
			theta = i*angle

			# Calculate the (x, y) coordinates of this vertex
			x = GCM[0] + ra*np.cos(theta)
			y = GCM[1] + ra*np.sin(theta)

			dg_p[0].append([x, y])
	else:
		for key, value in cluster_set.items():

			m = len(value); j = 1
			delta = 80*(np.pi/180)
			df = (GCM - cluster_center[key]); df_n = df/np.linalg.norm(df)
			angle = np.arctan2(df_n[1], df_n[0])
			r = ra*np.sqrt(len(value))
			# r = 4*np.cos(20*(np.pi/180))

			if angle < 0:

				angle += 2 * np.pi

			for i in range(m):

				if m == 1:

					j = 0
				else:

					j = ( (2*(i+1)-m-1)/(2*m-2) )

				theta = delta*j + np.pi + angle
				d = cluster_center[key] + np.array([r*np.cos(theta),\
													r*np.sin(theta)])
				dg_p[key].append(d.tolist())

			# print("m: ", m)
			# print("df: ", df)
			# print("Af: ", Af)
			# print("angle: ", angle)
			# print("dg_p: ", dg_p)
	# print("dg_p: ", dg_p)
	herding_computing_time_1 = time() - harding_starting_time_1;
	print("herding_computing_time_1: ", herding_computing_time_1)

	return dg_p, cluster_set, col_ind, CT

def K_means(targets, cameras):

	# Target Position
	data = np.array([target[0] for target in targets])

	# Define initial centers
	alpha = 0.0
	init_centers = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])

	# Create a KMeans instance with n clusters and defined initial centers
	kmeans = KMeans(n_clusters=len(cameras), init=init_centers, n_init=1)
	# kmeans = KMeans(n_clusters=len(cameras), n_init=1)

	# Fit the model to your data
	kmeans.fit(data)

	# Now you can get the cluster centers
	centers = kmeans.cluster_centers_
	print("Cluster centers:", centers)

	# And predict the cluster index for each sample
	labels = kmeans.predict(data)
	# print("Labels:", labels)

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(init_centers[:, np.newaxis] - centers, axis=2)
	# print("distance matrix: ", distance_matrix)
	row_ind, col_ind = linear_sum_assignment(distance_matrix)
	# print("row_ind, col_ind: ", row_ind, col_ind)

	return centers, col_ind

def kmeans_(targets, cameras):

	# Target Position
	data = np.array([target[0] for target in targets])
	# print("data: \n", data)

	# Define initial centers
	alpha = 0.0
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])
	init_centers = agents_position
	# init_centers = data[np.random.choice(len(data), len(cameras), replace=False)]

	max_iter = 1000
	tol = 1e-4

	# Cluster Computing Time
	CT = []
	cluster_starting_time = time()

	# print("init_centers: \n", init_centers)
	for _ in range(max_iter):

		distances = np.linalg.norm(data[:, np.newaxis] - init_centers, axis=2)
		# print("distances: ", distances)
		clusters = np.argmin(distances, axis=1)
		# print("clusters: ", clusters)
		row_ind, col_ind = linear_sum_assignment(distances); clusters = col_ind
		# print("col_ind: ", col_ind)
		new_centroids = np.array([data[clusters == j].mean(axis=0) for j in range(len(cameras))])
		# print("new_centroids: ", new_centroids)

		if np.linalg.norm(new_centroids - init_centers) < tol:

			break

		init_centers = new_centroids
	# print("init_centers: \n", init_centers)

	cluster_computing_time = time() - cluster_starting_time; CT.append(cluster_computing_time)

	# Cluster Computing Time
	allocation_starting_time = time()

	# Calculate the distance matrix
	# distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - data, axis=2)
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - init_centers, axis=2)
	# print("distance matrix: ", distance_matrix)
	row_ind, col_ind = linear_sum_assignment(distance_matrix)
	# print("row_ind, col_ind: ", row_ind, col_ind)

	allocation_computing_time = time() - allocation_starting_time; CT.append(allocation_computing_time)

	return CT, init_centers, col_ind

def FuzzyCMeans(targets, cameras):

	# Target Position
	targets_position = np.array([target[0] for target in targets])

	# Agent Position
	alpha = 0.0
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])
	c = len(agents_position)

	# Fuzzy C-Means algorithm
	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(targets_position.T, c, 2.0, error = 0.001, maxiter = 1000, seed = 42)
	cluster_centers = cntr
	cluster_membership = np.argmax(u, axis = 0)  # Get the cluster membership for each point
	# print("cluster_centers: ", cluster_centers)
	# print("cluster_membership: ", cluster_membership)

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_centers, axis=2)
	# print("distance matrix: ", distance_matrix)
	row_ind, col_ind = linear_sum_assignment(distance_matrix)
	# print("row_ind, col_ind: ", row_ind, col_ind)

	return cluster_centers, col_ind

def fuzzy_cmeans(targets, cameras):

	# Target Position
	targets_position = np.array([target[0] for target in targets])

	# Agent Position
	alpha = 0.0
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])

	n_samples = len(targets_position)
	n_features = 2
	# print("n_samples, n_features: ", n_samples, n_features)
	
	c = len(agents_position)
	U = np.random.dirichlet(np.ones(c), size=n_samples)
	# centroids = np.zeros((c, n_features))
	centroids = agents_position
	# print("U: \n", U)
	# print("centroids: \n", centroids)
	
	max_iter = 1000
	tol = 1e-4
	m = 2

	# Cluster Computing Time
	CT = []
	cluster_starting_time = time()

	for _ in range(max_iter):

		U_m = U ** m
		centroids = (U_m.T @ targets_position) / U_m.sum(axis=0)[:, None]
		distances = np.linalg.norm(targets_position[:, None] - centroids, axis=2)
		distances = np.fmax(distances, np.finfo(np.float64).eps)
		new_U = 1 / distances
		new_U = new_U / new_U.sum(axis=1, keepdims=True)

		if np.linalg.norm(new_U - U) < tol:

			break

		U = new_U
	print("centroids: ", centroids)

	cluster_computing_time = time() - cluster_starting_time; CT.append(cluster_computing_time)

	# Cluster Computing Time
	allocation_starting_time = time()

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - centroids, axis=2)
	# print("distance matrix: ", distance_matrix)
	row_ind, col_ind = linear_sum_assignment(distance_matrix)
	# print("row_ind, col_ind: ", row_ind, col_ind)

	allocation_computing_time = time() - allocation_starting_time; CT.append(allocation_computing_time)

	return CT

def HA(cluster_center, agents_position):

	# print("cluster_center: \n", cluster_center)
	# print("agents_position: \n", agents_position)

	agents_len = len(agents_position)
	targets_len = len(cluster_center)
	# print("agents_len: ", agents_len)
	# print("targets_len: ", targets_len)

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - cluster_center, axis=2)
	# print("distance_matrix: \n", distance_matrix)

	row_ind, col_ind = None, None
	if targets_len > agents_len or targets_len == agents_len:
		
		row_ind, col_ind = linear_sum_assignment(distance_matrix)
		# print("col_ind: ", col_ind)
		missing_numbers = []
	elif targets_len < agents_len:

		row_ind, col_ind = linear_sum_assignment(distance_matrix)

		col_sol = {str(i): [] for i in range(agents_len)}

		for (row, col) in zip(row_ind, col_ind):

			col_sol[str(row)] = col

		# List with one missing number
		sequence_num = list(range(0, agents_len))
		missing_numbers = [num for num in sequence_num if num not in row_ind]
		# print("missing_number: ", missing_numbers)

	return row_ind, col_ind, missing_numbers

delta = 0
MinPts = 2
Eps = 0
MinEps = 0
MaxEps = 0
signal = 1
phi = 0
bound_count = 0
ANOT_ = 0
ANOOT = 0
def DBSCAN_(targets, cameras, run_step, filename):

	global delta, MinPts, signal, phi, Eps, MaxEps, MinEps, bound_count, ANOT_, ANOOT
	print("delta, MinPts, signal, phi, Eps, MaxEps, MinEps: ", delta, MinPts, signal, phi, Eps, MaxEps, MinEps)

	# Self-Tuning of Parameters function
	ANOT = []
	Turning = True
	for camera in cameras:

		if camera.ANOT is None:

			Turning = False
			break
		else:

			ANOT.append(camera.ANOT)

	if Turning:

		# Perform logical AND on each column
		# print("ANOT: ", ANOT)
		column_and = np.logical_or.reduce(ANOT, axis=0)
		# print("column_and: ", column_and)

		# Converting boolean array to integer for display
		column_and = column_and.astype(int)
		# print("column_and: ", column_and)
		ANOT = float(np.sum(column_and))/float(len(targets))
		print("ANOT: ", ANOT)

		ANOT_ += ANOT*8
		ANOOT = np.round(ANOT_/run_step/8, 3)
		print("ANOOT: ", ANOOT)

		if (run_step%5 == 0):
		# if (run_step%10 == 0):

			# if (ANOOT - delta) < 0:

			# 	signal *= -1

			# if (ANOOT - delta) < 0:

			# 	signal = -1
			# 	bound_count = 0
			# elif (ANOOT - delta) > 0:

			# 	signal = 1
			# 	bound_count = 0
			# else:

			# 	bound_count += 1

			# 	if bound_count == 10:

			# 		signal *= -1
			# 		bound_count = 0

			Eps += phi*signal

			if Eps <= MinEps:

				Eps = MinEps

			elif Eps >= MaxEps:

				Eps = MaxEps

			delta = ANOOT

	# with open(filename, "a", encoding='UTF8', newline='') as f:

	# 	row = [Eps]
	# 	writer = csv.writer(f)
	# 	writer.writerow(row)

	# if Turning:

	# 	ANOT = np.array(ANOT)
	# 	# Calculate the number of `1`s in each column
	# 	count_ones = np.sum(ANOT == 1, axis=0)
	# 	print("count_ones: ", count_ones)

	# 	# Calculate the amount of `1`s, `2`s, and the number of elements >= `3`
	# 	num_ones = np.sum(count_ones >= 1)
	# 	num_twos = np.sum(count_ones >= 2)
	# 	num_threes_or_more = np.sum(count_ones >= 3)

	# 	KCOV = (num_ones+num_twos+num_threes_or_more)/len(targets)/3
	# 	print("KCOV: ", KCOV)

	# 	if (KCOV - delta) <= 0:

	# 		signal = -1
	# 	else:

	# 		signal = 1

	# 	MinEps += phi*signal

	# 	if MinEps <= phi:

	# 		MinEps = 0.14369104954571918
	# 	delta = KCOV

	self_center, self_label = [[] for _ in range(len(cameras))], [i for i in range(len(cameras))]

	X = np.array([target[0] for target in targets])

	# Compute DBSCAN
	db = DBSCAN(eps=Eps, min_samples=MinPts).fit(X)  # Adjust the eps and min_samples as needed
	labels = db.labels_
	# print("labels; ", labels)

	# Number of clusters in labels, ignoring noise if present
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	# print('Estimated number of clusters: %d' % n_clusters_)
	# print('Estimated number of noise points: %d' % n_noise_)

	# Step 1 - Find centroids of each cluster
	centroids = []
	for k in range(n_clusters_):

		class_member_mask = (labels == k)
		cluster_points = X[class_member_mask]
		centroid = cluster_points.mean(axis=0)
		centroids.append(centroid)

	centroids = np.array(centroids)
	print("Centroids of clusters:", centroids)

	# Step 2 - Assigned for the closest observer for each point
	alpha = 0.0
	if len(centroids) > 0:

		agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])

		row_ind, col_ind, missing_numbers = HA(centroids, agents_position)
		# print("row_ind: ", row_ind)
		# print("col_ind: ", col_ind)
		# print("missing_numbers: ", missing_numbers)

		for i, value in enumerate(row_ind):

			self_center[value] = centroids[col_ind[i]]
		# print("self_center: ", self_center)
	else:

		missing_numbers = list(range(0, len(cameras)))

	# Step 3 - Find indices of noise points
	noise_indices = np.where(labels == -1)[0]
	# print("Indices of noise points:", noise_indices)
	# print(halt)

	cameras_ = [cameras[i] for i in missing_numbers]
	missing_numbers_ = []
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras_])

	if (len(noise_indices) >= len(cameras_)) and (len(noise_indices) > 0) and (len(cameras_) > 0):

		targets_ = [targets[i] for i in noise_indices]
		# print("targets_: ", targets_)

		centers, labels = K_means(targets_, cameras_)
		# print("centers: ", centers)
		# print("labels: ", labels)

		row_ind, col_ind, missing_numbers_ = HA(centers, agents_position)
		# print("row_ind: ", row_ind)
		# print("col_ind: ", col_ind)
		# print("missing_numbers_: ", missing_numbers_)

		for i, value in enumerate(row_ind):

			self_center[missing_numbers[value]] = centers[col_ind[i]]
	elif (len(noise_indices) < len(cameras_)) and (len(noise_indices) > 0) and (len(cameras_) > 0):

		targets_ = [targets[i][0] for i in noise_indices]
		# print("targets_: ", targets_)

		row_ind, col_ind, missing_numbers_ = HA(targets_, agents_position)
		# print("row_ind: ", row_ind)
		# print("col_ind: ", col_ind)
		# print("missing_numbers_: ", missing_numbers_)

		for i, value in enumerate(row_ind):

			self_center[missing_numbers[value]] = targets_[col_ind[i]]
	elif (len(noise_indices) == 0) or (len(cameras_) == 0):

		missing_numbers_ = np.arange(0, len(missing_numbers))

	# print("self_center: ", self_center)
	# print(halt)

	# Step 4 - Residual observer assignment
	if len(missing_numbers_) > 0:

		cameras_ = [cameras[i] for i in np.array(missing_numbers)[missing_numbers_]]
		agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras_])
		targets_ = [targets[i][0] for i in range(len(targets))]
		row_ind, col_ind, missing_numbers__ = HA(targets_, agents_position)

		for i, value in enumerate(row_ind):

			self_center[np.array(missing_numbers)[missing_numbers_[value]]] = targets[col_ind[i]][0]

	self_center = np.array(self_center)
	print("self_center: \n", self_center)

	return self_center, self_label

def dbscan(targets, cameras, eps, min_samples):

	# Cluster Computing Time
	CT = []
	cluster_starting_time = time()

	X = np.array([target[0] for target in targets])

	n_samples = X.shape[0]
	labels = -np.ones(n_samples)
	cluster_id = 0

	def region_query(point_idx):

		distances = np.linalg.norm(X - X[point_idx], axis=1)

		return np.where(distances <= eps)[0]

	def expand_cluster(point_idx, neighbors):

		labels[point_idx] = cluster_id
		i = 0

		while i < len(neighbors):

			neighbor_idx = neighbors[i]

			if labels[neighbor_idx] == -1:

				labels[neighbor_idx] = cluster_id
			elif labels[neighbor_idx] == 0:

				labels[neighbor_idx] = cluster_id
				new_neighbors = region_query(neighbor_idx)

				if len(new_neighbors) >= min_samples:

					neighbors = np.append(neighbors, new_neighbors)

			i += 1

	for point_idx in range(n_samples):

		if labels[point_idx] == -1:

			neighbors = region_query(point_idx)

		if len(neighbors) >= min_samples:

			cluster_id += 1
			expand_cluster(point_idx, neighbors)
		else:

			labels[point_idx] = 0
	# print("neighbors: ", neighbors)

	cluster_computing_time = time() - cluster_starting_time; CT.append(cluster_computing_time)


	# Define initial centers
	alpha = 0.0
	agents_position = np.array([alpha*camera.pos + (1-alpha)*camera.sweet_spot for camera in cameras])

	# Cluster Computing Time
	allocation_starting_time = time()

	# Calculate the distance matrix
	distance_matrix = np.linalg.norm(agents_position[:, np.newaxis] - X, axis=2)
	# print("distance matrix: ", distance_matrix)
	row_ind, col_ind = linear_sum_assignment(distance_matrix)
	# print("row_ind, col_ind: ", row_ind, col_ind)

	allocation_computing_time = time() - allocation_starting_time; CT.append(allocation_computing_time)

	return CT

if __name__ == "__main__":

	for i in range(1):

		sleep(1)

		pygame.init()

		map_size = np.array([25, 25])
		grid_size = np.array([0.1, 0.1])

		cameras = []
		cameras_pos = []

		sensing_range = "2"

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
		# position_ = np.array([(2.0, 6.0), (15.8, 2.0), (15.0, 20.0)])
		# heading_ = np.array([(0.5, 0.0), (-0.55, 0.6), (0.0, -0.5)])

		# Extension
		# position_ = np.array([(2.0, 2.0), (23.0, 2.0), (2.0, 23.0), (23.0, 23.0), (12.5, 2.0)])
		# heading_ = np.array([(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (0.0, 1.0)])

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
		# 			'perspective'   :  np.array([0.0, 1.0]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (255, 250, 0)}
		# cameras.append(camera4)

		# camera5 = { 'id'            :  5,
		# 			'position'      :  np.array([23.0, 12.5]),
		# 			'perspective'   :  np.array([-1.0, 0.0]),
		# 			'AngleofView'   :  20,
		# 			'range_limit'   :  4,
		# 			'lambda'        :  2,
		# 			'color'         : (0, 240, 255)}
		# cameras.append(camera5)

		# camera6 = { 'id'            :  6,
		# 			'position'      :  np.array([12.5, 23.0]),
		# 			'perspective'   :  np.array([0.0, -1.0]),
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
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(14.5, 14.5), 1, 10], [(14.5, 13.5), 1, 10], [(19.5, 19.5), 1, 10], [(20.5, 20.5), 1, 10]]

		# Generate 8 random points
		# One
		# seed_key = {1: "103", 2: "106", 3: "143", 4: "279", 5: "351", 6: "333", 7: "555", 8: "913", 9: "3821", 10: "5214",
		# 			11: "5232", 12: "5246", 13: "5532"}
		# seed_key = {1: "5232", 2: "259", 3: "170", 4: "5688", 5: "333", 6: "33", 7: "555", 8: "6", 9: "3821", 10: "5214",
		# 			11: "5232", 12: "5246", 13: "5532"}

		# Two
		# seed_key = {1: "732", 2: "895", 3: "441", 4: "678", 5: "942", 6: "126", 7: "357", 8: "489", 9: "9021", 10: "6782",
		# 			11: "5432", 12: "1246", 13: "9876"}
		# seed_key = {1: "732", 2: "985", 3: "441", 4: "678", 5: "942", 6: "126", 7: "357", 8: "489", 9: "9021", 10: "6782",
		# 			11: "5432", 12: "1246", 13: "9876"}

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

		# Extend Scenario
		# seed_key = {'1': 8569, '2': 4370, '3': 2637, '4': 4853, '5': 3110, '6': 4000, '7': 4581, '8': 1770, '9': 4836, '10': 2987,
		# 			'11': 1619, '12': 5988, '13': 4155, '14': 4783, '15': 1334, '16': 1034, '17': 5790, '18': 4960, '19': 3033, '20': 3010}
		# seed_key = {'1': 5205, '2': 4000, '3': 5457, '4': 2810, '5': 1966, '6': 5407, '7': 5762, '8': 2330, '9': 5215, '10': 3310, 
		# 			'11': 4891, '12': 1024, '13': 3198, '14': 1567, '15': 5546, '16': 3304, '17': 2522, '18': 3288, '19': 3504, '20': 1166}
		# seed_key = {'1': 8402, '2': 8840, '3': 8906, '4': 9162, '5': 6996, '6': 6431, '7': 661, '8': 1828, '9': 4272, '10': 6335, 
		# 			'11': 6292, '12': 9971, '13': 7553, '14': 793, '15': 3126, '16': 9963, '17': 7819, '18': 2767, '19': 2644, '20': 9958}
		# seed_key = {'1': 7965, '2': 997, '3': 6213, '4': 9836, '5': 2180, '6': 968, '7': 7815, '8': 4654, '9': 5036, '10': 9269, 
		# 			'11': 9220, '12': 9598, '13': 7889, '14': 651, '15': 8315, '16': 9588, '17': 6272, '18': 5615, '19': 3161, '20': 6880}
		# seed_key = {'1': 3597, '2': 9651, '3': 4223, '4': 9165, '5': 6264, '6': 1647, '7': 4566, '8': 2081, '9': 4598, '10': 9372, 
		# 			'11': 5742, '12': 5949, '13': 5144, '14': 2655, '15': 6826, '16': 5158, '17': 4809, '18': 497, '19': 3297, '20': 2009}
		# seed_key = {'1': 6533, '2': 263, '3': 1204, '4': 5916, '5': 2853, '6': 2640, '7': 3658, '8': 8410, '9': 9250, '10': 7414, 
		# 			'11': 290, '12': 8439, '13': 7571, '14': 7930, '15': 9174, '16': 9236, '17': 7678, '18': 8429, '19': 7490, '20': 4943}
		# seed_key = {'1': 4956, '2': 4079, '3': 7792, '4': 9840, '5': 2972, '6': 6455, '7': 7490, '8': 9423, '9': 453, '10': 2865, 
		# 			'11': 6405, '12': 2172, '13': 9122, '14': 8124, '15': 9601, '16': 5717, '17': 2517, '18': 3077, '19': 6325, '20': 2443}
		# seed_key = {'1': 5800, '2': 4366, '3': 7090, '4': 1386, '5': 5318, '6': 3985, '7': 2100, '8': 660, '9': 2279, '10': 3353, 
		# 			'11': 8002, '12': 5922, '13': 7550, '14': 5100, '15': 6615, '16': 5495, '17': 9269, '18': 2831, '19': 553, '20': 9467}
		# seed_key = {'1': 4013, '2': 1407, '3': 238, '4': 6259, '5': 7890, '6': 7786, '7': 2575, '8': 9335, '9': 2034, '10': 1592, 
		# 			'11': 2420, '12': 5413, '13': 8719, '14': 6618, '15': 8713, '16': 5476, '17': 955, '18': 2542, '19': 7106, '20': 2659}
		seed_key = {'1': 6223, '2': 6122, '3': 6612, '4': 6723, '5': 2919, '6': 9584, '7': 1002, '8': 8797, '9': 8599, '10': 4176, 
					'11': 7208, '12': 7650, '13': 6979, '14': 3922, '15': 4681, '16': 5619, '17': 4827, '18': 8458, '19': 7506, '20': 2837}

		# np.random.seed(seed)
		# num_points = 8
		# x_coordinates = np.random.uniform(2, 23, num_points)
		# y_coordinates = np.random.uniform(2, 23, num_points)

		# targets = []
		# for i in range(num_points):

		# 	targets.append([(x_coordinates[i], y_coordinates[i]), 1, 10])

		# Normal Usage
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10]]

		# Empty Cluster
		# targets = [[(7.0, 9.0), 1, 10], [(11.5, 15.0), 1, 10], [(17.0, 9.0), 1, 10]]

		# # Empty Observation
		# targets = [[(12.5, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.5, 13.0), 1, 10]]

		# Excessive Assignment
		# targets = [[(12.5, 12.0), 1, 10], [(13.0, 13.0), 1, 10], [(13.5, 12.0), 1, 10], [(14.5, 10.5), 1, 10],
		# 			[(16.2, 9.5), 1, 10], [(11.9, 10.0), 1, 10], [(11.0, 11.5), 1, 10]]
		# targets = [[(12.5, 12.0), 1, 10], [(13.0, 13.0), 1, 10], [(13.5, 12.0), 1, 10], [(14.5, 10.0), 1, 10],
		# 			[(16.2, 9.5), 1, 10], [(11.9, 10.0), 1, 10], [(11.0, 11.5), 1, 10]]

		# DBSCAN
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(12.7, 12.7), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 12.5), 1, 10]]

		# Extension Usage
		# 4
		# targets = [[(12.0, 12.5), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10]]
		# targets = [[(14.4, 17.1), 1, 10], [(4.0, 14.5), 1, 10], [(11.6, 5.6), 1, 10], [(5.3, 19.4), 1, 10]]
		# targets = [[(20.3, 5.4), 1, 10], [(2.5, 8.9), 1, 10], [(10.6, 5.3), 1, 10], [(21.2, 14.5), 1, 10]]
		# targets = [[(19.7, 8.6), 1, 10], [(6.1, 4.4), 1, 10], [(13.9, 13.8), 1, 10], [(2.3, 17.9), 1, 10]]
		# targets = [[(8.2, 23.0), 1, 10], [(12.0, 21.9), 1, 10], [(15.7, 6.1), 1, 10], [(22.3, 16.7), 1, 10]]
		# targets = [[(9.66, 4.51), 1, 10], [(5.93, 14.26), 1, 10], [(21.54, 9.78), 1, 10], [(21.83, 17.47), 1, 10]]
		# targets = [[(15.98, 8.28), 1, 10], [(19.63, 15.46), 1, 10], [(13.04, 13.83), 1, 10], [(16.46, 6.64), 1, 10]]
		# targets = [[(10.54, 10.79), 1, 10], [(11.93, 12.01), 1, 10], [(14.7, 12.06), 1, 10], [(18.29, 7.06), 1, 10]]
		# targets = [[(21.85, 4.35), 1, 10], [(21.75, 21.92), 1, 10], [(18.82, 12.47), 1, 10], [(11.17, 4.54), 1, 10]]
		# targets = [[(4.64, 18.86), 1, 10], [(14.71, 5.3), 1, 10], [(12.61, 19.68), 1, 10], [(13.02, 10.58), 1, 10]]

		# 5
		# targets = [[(5.0, 5.0), 1, 10], [(6.0, 6.0), 1, 10], [(14.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(15.0, 12.5), 1, 10]]
		# targets = [[(4.18, 22.17), 1, 10], [(16.61, 3.97), 1, 10], [(12.29, 13.99), 1, 10], [(6.55, 21.81), 1, 10],
		# 			[(17.85, 19.42), 1, 10]]
		# targets = [[(4.96, 18.7), 1, 10], [(14.78, 15.09), 1, 10], [(5.56, 21.94), 1, 10], [(13.73, 7.74), 1, 10],
		# 			[(14.27, 11.24), 1, 10]]
		# targets = [[(12.2, 14.63), 1, 10], [(12.87, 21.61), 1, 10], [(7.08, 14.32), 1, 10], [(19.91, 18.72), 1, 10],
		# 			[(14.01, 6.76), 1, 10]]
		# targets = [[(10.25, 5.3), 1, 10], [(4.47, 6.68), 1, 10], [(4.77, 11.23), 1, 10], [(11.34, 21.96), 1, 10],
		# 			[(7.21, 21.17), 1, 10]]
		# targets = [[(21.82, 11.72), 1, 10], [(11.11, 13.53), 1, 10], [(19.25, 14.29), 1, 10], [(10.8, 16.54), 1, 10], 
		# 			[(7.08, 21.68), 1, 10]]
		# targets = [[(8.92, 18.16), 1, 10], [(14.23, 20.4), 1, 10], [(8.1, 22.0), 1, 10], [(20.44, 7.51), 1, 10], 
		# 			[(12.81, 5.86), 1, 10]]
		# targets = [[(10.9, 13.06), 1, 10], [(18.45, 20.45), 1, 10], [(20.96, 3.06), 1, 10], [(20.04, 11.8), 1, 10], 
		# 			[(5.31, 11.62), 1, 10]]
		# targets = [[(20.77, 22.34), 1, 10], [(4.1, 6.28), 1, 10], [(5.38, 18.62), 1, 10], [(10.3, 10.21), 1, 10], 
		# 			[(12.64, 8.99), 1, 10]]
		# targets = [[(4.05, 18.71), 1, 10], [(17.52, 7.69), 1, 10], [(21.45, 18.75), 1, 10], [(22.06, 10.38), 1, 10], 
		# 			[(15.75, 15.84), 1, 10]]

		# 6
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10]]
		# targets = [[(20.46, 8.39), 1, 10], [(4.85, 19.37), 1, 10], [(6.30, 4.15), 1, 10], [(11.96, 6.73), 1, 10], 
		# 			[(22.32, 3.85), 1, 10], [(17.63, 5.26), 1, 10]]
		# targets = [[(3.08, 5.33), 1, 10], [(19.78, 15.52), 1, 10], [(15.59, 6.22), 1, 10], [(2.86, 10.73), 1, 10], 
		# 			[(21.26, 3.19), 1, 10], [(5.16, 15.47), 1, 10]]
		# targets = [[(5.20, 14.43), 1, 10], [(7.08, 18.67), 1, 10], [(21.61, 7.58), 1, 10], [(14.28, 5.35), 1, 10], 
		# 			[(19.01, 8.86), 1, 10], [(10.62, 6.44), 1, 10]]
		# targets = [[(22.88, 21.68), 1, 10], [(4.82, 12.63), 1, 10], [(12.38, 2.50), 1, 10], [(8.41, 10.34), 1, 10], 
		# 			[(13.99, 22.12), 1, 10], [(19.37, 20.96), 1, 10]]
		# targets = [[(20.96, 13.5), 1, 10], [(14.96, 6.53), 1, 10], [(12.14, 11.62), 1, 10], [(3.94, 11.53), 1, 10], 
		# 			[(11.46, 17.96), 1, 10], [(13.56, 6.5), 1, 10]]
		# targets = [[(5.8, 9.94), 1, 10], [(18.22, 16.79), 1, 10], [(21.48, 13.93), 1, 10], [(12.27, 9.85), 1, 10], 
		# 			[(13.25, 18.43), 1, 10], [(17.72, 4.33), 1, 10]]
		# targets = [[(10.3, 16.12), 1, 10], [(14.7, 19.34), 1, 10], [(3.44, 21.3), 1, 10], [(4.91, 4.88), 1, 10], 
		# 			[(21.36, 19.54), 1, 10], [(6.86, 11.39), 1, 10]]
		# targets = [[(10.0, 21.01), 1, 10], [(14.88, 8.16), 1, 10], [(12.98, 10.67), 1, 10], [(10.46, 16.31), 1, 10], 
		# 			[(11.97, 8.83), 1, 10], [(19.14, 19.32), 1, 10]]
		# targets = [[(14.68, 3.21), 1, 10], [(6.34, 14.2), 1, 10], [(12.12, 16.45), 1, 10], [(13.25, 8.55), 1, 10], 
		# 			[(3.35, 7.64), 1, 10], [(17.25, 21.74), 1, 10]]

		# 7
		# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
		# 			[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10]]
		# targets = [[(20.67, 8.37), 1, 10], [(16.99, 7.56), 1, 10], [(16.13, 19.31), 1, 10], [(4.46, 15.07), 1, 10], 
		# 			[(13.22, 14.13), 1, 10], [(8.41, 2.58), 1, 10], [(4.95, 11.97), 1, 10]]
		# targets = [[(18.32, 10.97), 1, 10], [(15.47, 22.99), 1, 10], [(14.24, 11.56), 1, 10], [(7.85, 20.41), 1, 10], 
		# 			[(20.84, 7.97), 1, 10], [(7.48, 14.71), 1, 10], [(17.34, 17.53), 1, 10]]
		# targets = [[(18.94, 5.73), 1, 10], [(10.99, 13.28), 1, 10], [(15.94, 3.97), 1, 10], [(8.26, 11.25), 1, 10], 
		# 			[(9.97, 17.32), 1, 10], [(18.82, 6.45), 1, 10], [(14.32, 15.92), 1, 10]]
		# targets = [[(5.72, 9.76), 1, 10], [(3.72, 17.79), 1, 10], [(11.12, 7.98), 1, 10], [(6.45, 15.02), 1, 10], 
		# 			[(7.83, 11.16), 1, 10], [(13.72, 4.89), 1, 10], [(18.95, 20.91), 1, 10]]
		# targets = [[(17.28, 22.61), 1, 10], [(22.03, 19.05), 1, 10], [(22.08, 21.35), 1, 10], [(6.19, 6.1), 1, 10], 
		# 			[(6.62, 22.55), 1, 10], [(12.05, 4.59), 1, 10], [(14.15, 11.09), 1, 10]]
		# targets = [[(8.64, 19.65), 1, 10], [(8.07, 7.02), 1, 10], [(21.49, 8.8), 1, 10], [(21.55, 13.6), 1, 10], 
		# 			[(22.76, 18.2), 1, 10], [(10.35, 14.5), 1, 10], [(19.34, 7.85), 1, 10]]
		# targets = [[(19.68, 5.62), 1, 10], [(12.05, 17.95), 1, 10], [(8.22, 4.94), 1, 10], [(8.96, 9.39), 1, 10], 
		# 			[(20.98, 22.41), 1, 10], [(20.45, 6.99), 1, 10], [(8.97, 7.18), 1, 10]]
		# targets = [[(5.7, 22.93), 1, 10], [(11.82, 20.38), 1, 10], [(11.41, 16.15), 1, 10], [(22.21, 8.48), 1, 10], 
		# 			[(18.71, 20.85), 1, 10], [(3.63, 22.23), 1, 10], [(14.82, 9.63), 1, 10]]
		# targets = [[(10.11, 4.34), 1, 10], [(18.8, 17.92), 1, 10], [(8.28, 3.44), 1, 10], [(16.55, 6.22), 1, 10], 
		# 			[(12.05, 21.51), 1, 10], [(4.12, 20.47), 1, 10], [(14.01, 16.72), 1, 10]]

		# 8
		targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
					[(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10]]
		# targets = [[(20.6, 4.3), 1, 10], [(14.8, 21.7), 1, 10], [(8.6, 7.4), 1, 10], [(5.3, 11.9), 1, 10], 
		# 			[(20.1, 8.7), 1, 10], [(15.2, 10.6), 1, 10], [(22.0, 13.6), 1, 10], [(9.7, 22.3), 1, 10]]
		# targets = [[(6.2, 18.0), 1, 10], [(13.5, 3.9), 1, 10], [(3.4, 15.6), 1, 10], [(21.3, 14.5), 1, 10], 
		# 			[(8.0, 6.5), 1, 10], [(5.2, 13.7), 1, 10], [(16.7, 12.2), 1, 10], [(9.4, 21.8), 1, 10]]
		# targets = [[(12.3, 22.2), 1, 10], [(5.9, 5.5), 1, 10], [(10.7, 10.1), 1, 10], [(16.8, 20.2), 1, 10], 
		# 			[(18.4, 16.7), 1, 10], [(22.6, 19.8), 1, 10], [(8.2, 15.3), 1, 10], [(17.6, 9.9), 1, 10]]
		# targets = [[(14.6, 3.2), 1, 10], [(19.0, 13.3), 1, 10], [(5.6, 17.7), 1, 10], [(16.8, 6.4), 1, 10], 
		# 			[(11.4, 9.2), 1, 10], [(7.6, 21.6), 1, 10], [(22.9, 12.7), 1, 10], [(9.1, 10.5), 1, 10]]
		# targets = [[(17.49, 12.11), 1, 10], [(20.65, 19.37), 1, 10], [(9.36, 20.87), 1, 10], [(20.18, 12.37), 1, 10],
		# 			[(19.48, 22.99), 1, 10], [(20.14, 7.31), 1, 10], [(14.35, 19.18), 1, 10], [(7.89, 6.04), 1, 10]]
		# targets = [[(7.68, 14.93), 1, 10], [(18.18, 14.84), 1, 10], [(22.30, 19.19), 1, 10], [(18.07, 4.26), 1, 10], 
		# 			[(14.13, 14.85), 1, 10], [(4.96, 3.66), 1, 10], [(18.80, 18.97), 1, 10], [(17.54, 19.04), 1, 10]]
		# targets = [[(15.03, 3.61), 1, 10], [(11.06, 16.42), 1, 10], [(8.95, 12.32), 1, 10], [(20.24, 20.45), 1, 10], 
		# 			[(10.18, 8.37), 1, 10], [(19.04, 7.89), 1, 10], [(15.03, 10.51), 1, 10], [(11.95, 18.73), 1, 10]]
		# targets = [[(7.73, 11.65), 1, 10], [(22.26, 7.34), 1, 10], [(4.12, 11.06), 1, 10], [(14.82, 16.68), 1, 10], 
		# 			[(12.72, 18.20), 1, 10], [(11.90, 11.52), 1, 10], [(4.22, 6.03), 1, 10], [(13.39, 8.46), 1, 10]]
		# targets = [[(10.75, 4.76), 1, 10], [(20.12, 16.44), 1, 10], [(9.35, 16.31), 1, 10], [(21.77, 4.92), 1, 10], 
		# 			[(3.92, 11.51), 1, 10], [(8.60, 19.28), 1, 10], [(7.76, 9.47), 1, 10], [(11.07, 3.74), 1, 10]]

		# targets = [[(7.68, 14.93), 1, 10], [(18.18, 10.84), 1, 10], [(22.30, 19.19), 1, 10], [(18.07, 4.26), 1, 10], 
		# 			[(14.13, 14.85), 1, 10], [(4.96, 3.66), 1, 10], [(18.80, 18.97), 1, 10], [(16.54, 19.04), 1, 10]]

		# Start Simulation
		Done = False
		vis = Visualize(map_size, grid_size)
		last = time()

		Times = "Ten/"
		Alg_key = {0: "K", 1: "DBSCAN", 2: "PA", 3: "FCM"}
		speed_gain = "0.90"
		save_key = 10
		save_type = {1: "Target_Speed/KC/", 2: "Sensing_Range/KC/", 3: "Target_Speed/APC/", 4: "Sensing_Range/APC/",
						5: speed_gain + "_" + sensing_range, 6: "kcov",
					7: "S_20/", 8: "S_60/", 9: "T_10/", 10: "T_90/"}

		velocities = (float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)  # Random initial velocities (-0.5 to 0.5)

		# cp = Alg_key[int(i)]
		# cp = "FCM"
		# cp = "K"
		# cp = "PA"
		cp = "DBSCAN"

		if cp == "DBSCAN":

			MaxEps = 9
			MinEps = 3
			Eps = 6
			phi = 0.1

			# Eps = 2.0*uav_team.members[0].incircle_r
			# MinEps = 0.1*uav_team.members[0].incircle_r
			# MaxEps = 4.0*uav_team.members[0].incircle_r
			# phi = 0.1*uav_team.members[0].incircle_r
		
		# for i in range(len(cameras)):

		# 	if cp == "FCM":

		# 		# ANOT
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/FCM/" + Times
		# 		# filename += save_type[save_key] + str(int(float(speed_gain)*100)) + "/FCM_" + str(i) + ".csv"
		# 		# filename += save_type[save_key] + str(int(float(sensing_range))) + "/FCM_" + str(i) + ".csv"

		# 		# KCOV
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/FCM/" + Times
		# 		# filename += "FCM_" + str(i) + "_" + save_type[save_key] + ".csv"

		# 		# CT
		# 		filename = "/home/leo/mts/src/QBSM/Extend/Data/CT/FCM/" + Times
		# 		filename += "FCM_CT.csv"
		# 	elif cp == "K":

		# 		# ANOT
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/K/" + Times
		# 		# filename += save_type[save_key] + str(int(float(speed_gain)*100)) + "/K_" + str(i) + ".csv"
		# 		# filename += save_type[save_key] + str(int(float(sensing_range))) + "/K_" + str(i) + ".csv"

		# 		# KCOV
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/K/" + Times
		# 		# filename += "K_" + str(i) + "_" + save_type[save_key] + ".csv"

		# 		# CT
		# 		filename = "/home/leo/mts/src/QBSM/Extend/Data/CT/K/" + Times
		# 		filename += "K_CT.csv"
		# 	elif cp == "PA":

		# 		# ANOT
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/PA/" + Times
		# 		# filename += save_type[save_key] + str(int(float(speed_gain)*100)) + "/PA_" + str(i) + ".csv"
		# 		# filename += save_type[save_key] + str(int(float(sensing_range))) + "/PA_" + str(i) + ".csv"

		# 		# KCOV
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/PA/" + Times
		# 		# filename += "PA_" + str(i) + "_" + save_type[save_key] + ".csv"

		# 		# CT
		# 		filename = "/home/leo/mts/src/QBSM/Extend/Data/CT/PA/" + Times
		# 		filename += "PA_CT.csv"
		# 	elif cp == "DBSCAN":

		# 		# ANOT
		# 		filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/DBSCAN/" + Times
		# 		# filename += save_type[save_key] + str(int(float(speed_gain)*100)) + "/DBSCAN_" + str(i) + ".csv"
		# 		filename += save_type[save_key] + str(int(float(sensing_range))) + "/DBSCAN_" + str(i) + ".csv"

		# 		# KCOV
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/KCOV/DBSCAN/" + Times
		# 		# filename += "DBSCAN_" + str(i) + "_" + save_type[save_key] + ".csv"

		# 		# CT
		# 		# filename = "/home/leo/mts/src/QBSM/Extend/Data/CT/DBSCAN/" + Times
		# 		# filename += "DBSCAN_CT.csv"

		# 	f = open(filename, "w+")
		# 	f.close()

		run_step = 1

		while not Done:

			overall_computing_time = []

			print("run_step: ", run_step)

			for op in pygame.event.get():

				if op.type == pygame.QUIT:

					Done = True

			# if run_step <= 900 and run_step >= 100:
			if run_step >= 100:

				# Simulation parameters
				time_step = 0.1  # Time step in seconds
				min_distance = 0.7  # Minimum distance between points to avoid collision
				boundary_margin = 2  # Minimum distance from the boundary
				tracker_margin = 1  # Minimum distance from the boundary

				# Initialize point positions and velocities
				positions = [target[0] for target in targets]
				# print("positions: " + str(positions))
				# velocities = np.random.rand(4, 2) - 0.5  # Random initial velocities (-0.5 to 0.5)
				# print("velocities: " + str(velocities))

				positions += velocities * time_step
				# print("positions: " + str(positions))

				# Change direction of velocities every 3 seconds
				# if run_step%100 == 0:
				if run_step%50 == 0:
					
					# np.random.seed(int(seed_key[int(run_step/100)]))
					# velocities = (float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)

					np.random.seed(seed_key[str(int(run_step/50))])
					velocities = (float(speed_gain)/0.5)*(np.random.rand(len(targets), 2) - 0.5)

					# v_ = np.array([(1,0), (0,-1), (-1,0)])
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

						# velocities[i, 0] *= -0.5  # Reverse x-direction velocity

						if abs(positions[i, 0] - 0) <= boundary_margin:

							velocities[i, 0] = (+1)*abs(velocities[i, 0]) # Reverse x-direction velocity
						elif abs(positions[i, 0] - 25) <= boundary_margin:

							velocities[i, 0] = (-1)*abs(velocities[i, 0]) # Reverse x-direction velocity
					if abs(positions[i, 1] - 0) <= boundary_margin or abs(positions[i, 1] - 25) <= boundary_margin:

						# velocities[i, 1] *= -0.5  # Reverse y-direction velocity

						if abs(positions[i, 1] - 0) <= boundary_margin:

							velocities[i, 1] = (+1)*abs(velocities[i, 0]) # Reverse x-direction velocity
						elif abs(positions[i, 1] - 25) <= boundary_margin:

							velocities[i, 1] = (-1)*abs(velocities[i, 0]) # Reverse x-direction velocity

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

				one_hop_neighbor, Team = None, None
				Pd, cluster_set, col_ind = None, None, None
				cluster_centers, cluster_labels = FuzzyCMeans(targets, uav_team.members)

				# CT_12 = fuzzy_cmeans(targets, uav_team.members)
				# CT_3 = One_hop_neighbor_n(points)

				# centralized_computing_time = [CT_12[0], CT_12[1], CT_3]
				# print("centralized " + cp +" computing_time: ", centralized_computing_time)
				# overall_computing_time.extend(centralized_computing_time)
			elif cp == "K":

				one_hop_neighbor, Team = None, None
				Pd, cluster_set, col_ind = None, None, None
				cluster_centers, cluster_labels = K_means(targets, uav_team.members)

				# CT_12, _, _ = kmeans_(targets, uav_team.members)
				# CT_3 = One_hop_neighbor_n(points)

				# centralized_computing_time = [CT_12[0], CT_12[1], CT_3]
				# print("centralized " + cp +" computing_time: ", centralized_computing_time)
				# overall_computing_time.extend(centralized_computing_time)
			elif cp == "DBSCAN":

				# filename = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/DBSCAN/" + Times
				# filename += save_type[save_key] + str(int(float(speed_gain)*100)) + "/Eps.csv"
				# filename += save_type[save_key] + str(int(float(sensing_range))) + "/Eps.csv"
				filename = "None"

				one_hop_neighbor, Team = None, None
				Pd, cluster_set, col_ind = None, None, None
				cluster_centers, cluster_labels = DBSCAN_(targets, uav_team.members, run_step, filename)

				# Centralized Starting Time
				# centralized_start_time = time()

				# CT_12 = dbscan(targets, uav_team.members, 2*uav_team.members[0].incircle_r, 2)
				# CT_34, _, _  = kmeans_(targets, uav_team.members)
				# CT_5 = One_hop_neighbor_n(points)
				
				# centralized_computing_time = [CT_12[0], CT_12[1], CT_34[0], CT_34[1], CT_5]
				# print("centralized " + cp +" computing_time: ", centralized_computing_time)
				# overall_computing_time.extend(centralized_computing_time)
			elif cp == "PA":

				# one_hop_neighbor = One_hop_neighbor(points)

				cluster_labels, cluster_centers = None, None
				# Pd, cluster_set, col_ind = Hungarian(targets, uav_team.members)
				# Pd, cluster_set, col_ind = BILP(targets, uav_team.members)
				# one_hop_neighbor = One_hop_neighbor_E(points, cluster_set, col_ind)

				dg_p, cluster_set, col_ind, CT_12 = Allocation(targets, uav_team.members)
				Pd, one_hop_neighbor, Team, CT_3 = One_hop_neighbor(points, cluster_set, col_ind, dg_p)
				# A = One_hop_neighbor_n(points)

				# centralized_computing_time = [CT_12[0], CT_12[1], CT_3]
				# print("centralized " + cp +" computing_time: ", centralized_computing_time)
				# overall_computing_time.extend(centralized_computing_time)

			for i in range(len(uav_team.members)):

				neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]

				# Decentralized Starting Time
				decentralized_start_time = time()

				uav_team.members[i].UpdateState(targets, neighbors, one_hop_neighbor, Pd, cluster_set, col_ind, cluster_labels, 
					cluster_centers, Team, velocities, np.round(time() - last, 2), cp, speed_gain, sensing_range, save_type[save_key], Times)

				decentralized_computing_time = time() - decentralized_start_time
				print("decentralized " + str(i) +" computing_time: ", decentralized_computing_time)
				overall_computing_time.append(decentralized_computing_time)

			# Computing Time
			# filename = "/home/leo/mts/src/QBSM/Extend/Data/CT/" + cp + "/" + Times
			# filename += cp + "_CT.csv"

			# with open(filename, "a", encoding='UTF8', newline='') as f:

			# 	row = overall_computing_time
			# 	writer = csv.writer(f)
			# 	writer.writerow(row)

			# Total Simulation Time
			print("Simulation Time: " + str(time() - last))

			circumcenter_center, circumcenter_radius = circumcenter(targets)
			side_center, side_center_radius = sidecenter(targets)

			vis.Visualize2D(uav_team.members, event_plt1, targets, circumcenter_center, circumcenter_radius, \
							side_center, side_center_radius, run_step, cp, Eps, ANOOT)

			# if np.round(time() - last, 2) > 60.00:
			if run_step >= 1000:

				# sys.exit()
				break

			run_step += 1

		pygame.quit()