import timeit
import numpy as np
import numexpr as ne
from math import sqrt, acos, cos
from matplotlib.path import Path
from scipy.integrate import quad
from shapely.geometry import Point
from collections import namedtuple
from scipy.optimize import linprog
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from collections import defaultdict

# from alphashape import alphashape
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, Delaunay
from gudhi import AlphaComplex
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift

from sklearn.cluster import spectral_clustering
from sklearn.metrics import pairwise_distances
from sklearn.cluster import OPTICS

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.cluster.vq import kmeans, vq

from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import fcluster
from sklearn.datasets import make_blobs
import skfuzzy as fuzz

a = np.arange(0, 30, 0.1)
b = np.arange(0, 30, 0.1)
X, Y = np.meshgrid(a, b)

W = np.vstack([X.ravel(), Y.ravel()])
W = W.transpose()

def HA():

	agents_len = 5
	targets_len = 3

	col_sol = {str(i): [] for i in range(agents_len)}

	hold = np.inf*np.ones([agents_len, agents_len*targets_len])
	# Sample data: reward matrix (m x n)
	cost_matrix = np.array([[1, 7, 11],
							[3, 11, 8],
							[4, 1, 8],
							[5, 6, 1],
							[7, 5, 10]])

	# for i in range(agents_len):

	# 	hold[i,i*targets_len:(i+1)*targets_len] = cost_matrix[i,:]

	# print("hold: ", hold)
	# cost_matrix = hold

	# Apply the Linear Assignment Algorithm to maximize the reward
	row_indices, col_indices = linear_sum_assignment(cost_matrix)
	print("row_indices: ", row_indices)
	print("col_indices: ", col_indices)

	# col_indices = col_indices%targets_len

	# Print the assignments
	for i, j in zip(row_indices, col_indices):

		print(f"Agent {i} assigned to Target {j}")

	for (row, col) in zip(row_indices, col_indices):

		col_sol[str(row)] = col

	# List with one missing number
	sequence_num = list(range(0, agents_len))
	missing_numbers = [num for num in sequence_num if num not in row_indices]

	print("missing_number: ", missing_numbers)

	cost_matrix_missing = np.array(cost_matrix[missing_numbers])
	print("cost_matrix_missing: ", cost_matrix_missing)

	row_indices_missing, col_indices_missing = linear_sum_assignment(cost_matrix_missing)
	print("row_indices_missing: ", row_indices_missing)
	print("col_indices_missing: ", col_indices_missing)

	for (row, col) in zip(missing_numbers, col_indices_missing):

		col_sol[str(row)] = col

	print("col_sol: ", col_sol)
	
	col_indices = np.zeros(agents_len)
	for key_, value_ in col_sol.items():

		col_indices[int(key_)] = value_

	# col_indices = np.hstack((col_indices, col_indices_missing))
	print("col_indices: ", col_indices)

def and_or_test():

	a = np.array([[10.5, 12.5]])
	b = np.array([[12, 12], [12, 13], [13, 12], [13, 13], [10.5,12.5]])
	print((a == b))
	print(np.all((a == b), axis = 0))
	print(np.all((a == b), axis = 1).any())

def npn():

	return np.linalg.norm(W)

def nen():

	x = W[:,0]; y = W[:,1]
	return ne.evaluate('sqrt(x**2 + y**2)')

def para_opt():

	number = 1

	x = timeit.timeit(stmt = "npn()", number = number, globals = globals())

	print(x)

	x = timeit.timeit(stmt = "nen()", number = number, globals = globals())

	print(x)

def circumcenter(x1, y1, x2, y2, x3, y3):

	d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
	ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
	uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

	return ux, uy, d

	# if (x2 - x1)*(y3 - y1) == (y2 - y1)*(x3 - x1):

	# 	avg_x = (x1 + x2 + x3)/3
	# 	avg_y = (y1 + y2 + y3)/3
	# 	geometric_center = np.array([avg_x, avg_y])
		
	# 	r = 0.0
	# 	points = np.array([(x1, y1), (x2, y2), (x3, y3)])

	# 	for i in range(len(points)):

	# 		dist = np.linalg.norm(geometric_center - points[i])

	# 		if dist >= r:

	# 			r = dist

	# 	center_x = avg_x
	# 	center_y = avg_y
	# 	radius = r
	# else:

	# 	mid_ab_x = (x1 + x2) / 2
	# 	mid_ab_y = (y1 + y2) / 2

	# 	mid_bc_x = (x2 + x3) / 2
	# 	mid_bc_y = (y2 + y3) / 2

	# 	slope_ab = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
	# 	slope_bc = (y3 - y2) / (x3 - x2) if x3 != x2 else float('inf')

	# 	if slope_ab == 0:

	# 		slope_bc = -float('inf')
	# 	elif slope_bc == 0:

	# 		slope_ab = -float('inf')
	# 	else:

	# 		slope_ab = -1 / slope_ab
	# 		slope_bc = -1 / slope_bc

	# 	center_x = (mid_bc_y - mid_ab_y + slope_ab * mid_ab_x - slope_bc * mid_bc_x) / (slope_ab - slope_bc)
	# 	center_y = mid_ab_y + slope_ab * (center_x - mid_ab_x)

	# 	radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5

	# return center_x, center_y, radius

def calculate_tangent_angle(circle_center, circle_radius, point):

	distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
	adjcent = np.sqrt(distance**2 - circle_radius**2)
	angle = 2*np.arctan(circle_radius/adjcent)

	# # Calculate the distance between the circle center and the point
	# distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)

	# # Calculate the length of the line segment connecting the circle center and the point
	# length = distance - circle_radius

	# # Calculate the angle using the arctan function
	# angle = 2*np.arctan(length / (2*circle_radius))

	return (angle*180)/np.pi

def hurg():

	cost_1 = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
	row_ind, col_ind = linear_sum_assignment(cost_1)

	print(row_ind, col_ind)

	for row, col in zip(row_ind, col_ind):

		print(f"Agent {row+1} assigned to Task {col+1}")

def name_tuple():

	Point = namedtuple("Coordinat", ["x", "y", "z"])
	point = Point(1, 2, 3)

	print(point.x)

def integrand():

	a = 0
	b = 1

	X = lambda x: x**2

	integral, error = quad(X, a, b)

	print(f"The integral of x^2 from {a} to {b} is {integral:.4f} with an error of {error:.4f}")

def linear_program():

	c = [-1, 4]
	A = [[-3, 1], [1, 2]]
	b = [6, 4]
	x0_bounds = (None, None)
	x1_bounds = (-3, None)

	res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])

	print(f"Optimal value: {res.fun:.4f}")
	print(f"Optimal solution: {res.x}")

def MST():

	targets = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]])

	# Calculate the pairwise distances between targets
	distances = distance.cdist(targets, targets)

	print("distances: " + str(distances))

	# Create a sparse adjacency matrix from the distances
	adj_matrix = csr_matrix(distances)

	print("adj_matrix: " + str(adj_matrix))

	# Compute the minimum spanning tree using Kruskal's algorithm
	mst = minimum_spanning_tree(adj_matrix)

	print("mst: " + str(mst))

	# Extract the edges from the MST
	edges = np.array(mst.nonzero()).T

	print("edges: " + str(edges))

	# Plot the targets and MST edges
	plt.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets')

	for edge in edges:

		start = targets[edge[0]]
		end = targets[edge[1]]
		plt.plot([start[0], end[0]], [start[1], end[1]], color='blue')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Minimum Spanning Tree of Targets')
	plt.legend()
	plt.show()

def MSF():

	# Generate example target coordinates
	targets = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [30, 30]])

	# Calculate the pairwise distances between targets
	distances = distance.cdist(targets, targets)

	# Create a sparse adjacency matrix from the distances
	adj_matrix = csr_matrix(distances)

	# Compute the minimum spanning tree using Kruskal's algorithm
	mst = minimum_spanning_tree(adj_matrix)

	# Convert the MST to a dense matrix
	mst_dense = mst.toarray().astype(bool)

	# Find the connected components in the MST
	n_components, labels = np.unique(mst_dense, return_inverse=True, axis=0)

	# Plot the targets and the minimum spanning forest
	plt.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets')

	for i in range(n_components.shape[0]):

		component_indices = np.where(labels == i)[0]
		component_targets = targets[component_indices]
		component_mst = mst_dense[component_indices][:, component_indices]

		for edge in np.transpose(component_mst.nonzero()):

			start = component_targets[edge[0]]
			end = component_targets[edge[1]]
			plt.plot([start[0], end[0]], [start[1], end[1]], color='blue')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Minimum Spanning Forest of Targets')
	plt.legend()
	plt.show()

def MST2MSF():

	targets = np.array([[1, 2], [3, 5], [5, 6], [7, 8], [9, 10], [31, 12], [30, 14]])

	# Calculate the pairwise distances between targets
	distances = distance.cdist(targets, targets)

	print("distances: " + "\n" + str(distances) + "\n")

	# Create a sparse adjacency matrix from the distances
	adj_matrix = csr_matrix(distances)

	print("adj_matrix: " + "\n" + str(adj_matrix) + "\n")

	# Compute the minimum spanning tree using Kruskal's algorithm
	mst = minimum_spanning_tree(adj_matrix)

	print("mst: " + "\n" + str(mst) + "\n")

	# Extract the edges from the MST
	edges = np.array(mst.nonzero()).T

	print("edges: " + "\n" + str(edges) + "\n")

	# Define the edges of the minimum spanning tree
	mst_edges = [tuple(edge) for edge in edges]

	print("mst_edges: " + "\n" + str(mst_edges) + "\n")

	# Define the weights of the edges in the minimum spanning tree
	mst_weights = [mst.toarray().astype(float)[index[0], index[1]] for index in mst_edges]

	print("mst_weights: " + "\n" + str(mst_weights) + "\n")

	# Define the weight threshold for deleting edges
	weight_threshold = 4

	modified_edges, modified_weights = [], []

	for edge, weight in zip(mst_edges, mst_weights):

		# Check if the weight of the edge exceeds the threshold
		if weight <= weight_threshold:

			# Add the edge to the modified minimum spanning tree
			modified_edges.append(edge)
			modified_weights.append(weight)

	# modified_edges = np.array(modified_edges)

	print("Edges of the Modified Minimum Spanning Tree: " + str(modified_edges) + "\n")
	print("modified_weights: " + str(modified_weights) + "\n")
	
	'''
	print("Check_list: ", end = '')
	print(1 == modified_edges)

	b = np.logical_or((1 == modified_edges)[:,0], (1 == modified_edges)[:,1])
	print("\n" + "Logical or: " + str(b) + "\n")
	
	c = np.where(b == True, modified_weights, np.inf)
	print("Weight: " + str(c) + "\n")

	d = modified_edges[(b == True)]
	print("Path: " + str(d) + "\n")

	e = d[np.argmin(c)]
	print("Right Way: " + str(e) + "\n")
	'''

	# Plot the targets and MST edges
	plt.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets')

	for edge in modified_edges:

		start = targets[edge[0]]
		end = targets[edge[1]]
		plt.plot([start[0], end[0]], [start[1], end[1]], color='blue')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Minimum Spanning Tree of Targets')
	plt.legend()
	plt.show()

def SEMST():

	start_vertex = 1
	targets = np.array([[1, 2], [3, 5], [5, 6], [7, 8], [9, 10], [31, 12], [30, 14]])

	# Calculate the pairwise distances between targets
	distances = distance.cdist(targets, targets)
	print("distances: " + "\n" + str(distances) + "\n")

	num_vertices = np.shape(distances[0])[0]
	print("num_vertices: " + str(num_vertices) + "\n")

	# visited = [False]*np.ones(num_vertices)
	# visited[start_vertex] = True

	# mst_edges = []
	# mst_weights = []
	# while len(mst_edges) < num_vertices - 1:

	# 	min_edge = None
	# 	min_weight = float('inf')

	# 	for i in range(num_vertices):
			
	# 		if visited[i]:
				
	# 			for j in range(num_vertices):

	# 				if not visited[j] and distances[i, j] < min_weight:
							
	# 					min_edge = (i, j)
	# 					min_weight = distances[i, j]
	# 	if min_edge:

	# 		mst_edges.append(min_edge)
	# 		mst_weights.append(min_weight)
	# 		visited[min_edge[1]] = True

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

					if (not visited[j]) and (distances[i, j] < min_weight):
							
						min_edge = (i, j)
						min_weight = distances[i, j]
		if min_edge:

			mst_edges.append(min_edge)
			mst_weights.append(min_weight)
			visited[min_edge[1]] = True
			temp_root = min_edge[1]

	print("MST: " + str(mst_edges) + "\n")
	print("Weights: " + str(mst_weights) + "\n")

	# Define the weight threshold for deleting edges
	weight_threshold = 4

	modified_edges, modified_weights = [], []

	for edge, weight in zip(mst_edges, mst_weights):

		# Check if the weight of the edge exceeds the threshold
		if weight <= weight_threshold:

			# Add the edge to the modified minimum spanning tree
			modified_edges.append(edge)
			modified_weights.append(weight)

	print("Modified MST: " + str(modified_edges) + "\n")
	print("Modified Weights: " + str(modified_weights) + "\n")

	plt.scatter(targets[:, 0], targets[:, 1], color='red', label='Targets')

	for edge in modified_edges:

		start = targets[edge[0]]
		end = targets[edge[1]]
		plt.plot([start[0], end[0]], [start[1], end[1]], color='blue')

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Minimum Spanning Tree of Targets')
	plt.legend()
	plt.show()

def delete_element_in_matrix():

	a = [(1,1), (2,2), (3,3), (4,4)]
	b = set([(1,1), (2,2)])
	c = [x for x in a if x not in b]

	print(c)

def test():

	my_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	element = np.array([4, 5, 6])

	# Check if the element is present in the array
	is_present = any(np.array_equal(element, arr) for arr in my_array)

	v = np.array([(11.5, 11.5), (11.5, 13.5)], dtype = float)
	d = np.array((13.5, 13.5), dtype = float)
	print(any(np.array_equal(d, arr) for arr in v))

	a = np.array([(1,5), (1,5), (1,5), (4,8), (5,9)])
	b = np.concatenate((a, a))
	distances = distance.cdist(b, b)
	print("distances: " + str(distances))

def concavehull():

	# Define the input points (x, y coordinates)
	points = [(1, 1), (2, 3), (4, 2), (15, 14), (16, 15), (17, 3)]

	# Compute the concave hull
	alpha = 0.5  # Alpha parameter (controls the concavity)
	concave_hull = alphashape(points, alpha)
	print("concave hull: " + str(concave_hull))

	# Plot the input points and the concave hull
	fig, ax = plt.subplots()
	ax.scatter(*zip(*points), color='red', label='Points')
	ax.plot(*zip(*concave_hull.exterior.coords), color='blue', label='Concave Hull')
	ax.set_aspect('equal', 'box')
	ax.legend()
	plt.show()

def del_tri():

	# Define the input points (x, y coordinates)
	# points = [(1, 1), (2, 3), (4, 2), (5, 4), (6, 1), (7, 3)]
	points = [(1, 1), (2, 3), (4, 2), (5, 4), (8, 8), (8, 3)]

	# Compute the concave hull
	alpha = 0.4 # Alpha parameter (controls the concavity)
	concave_hull = alphashape(points, alpha)

	print(type(concave_hull))

	if concave_hull.geom_type == 'MultiPolygon':

		mycoordslist = [list(x.exterior.coords) for x in concave_hull.geoms]
		print(mycoordslist)

		# Plot the input points and the concave hull
		fig, ax = plt.subplots()
		ax.scatter(*zip(*points), color='red', label='Points')
		ax.plot(*zip(*mycoordslist[0]), color='blue', label='Concave Hull')
		ax.plot(*zip(*mycoordslist[1]), color='blue')
		ax.set_aspect('equal', 'box')
		ax.legend()
		plt.show()
	elif concave_hull.geom_type == "Polygon":

		print([*concave_hull.exterior.coords])

		# Plot the input points and the concave hull
		fig, ax = plt.subplots()
		ax.scatter(*zip(*points), color='red', label='Points')
		ax.plot(*zip(*concave_hull.exterior.coords), color='blue', label='Concave Hull')
		ax.set_aspect('equal', 'box')
		ax.legend()
		plt.show()
	else:

		fig, ax = plt.subplots()
		ax.scatter(*zip(*points), color='red', label='Points')
		ax.set_aspect('equal', 'box')
		ax.legend()
		plt.show()

def alpha_complex():

	points = np.array([(1, 1), (2, 3), (4, 2), (5, 4), (8, 8), (8, 3)])
	alpha_complex = AlphaComplex(points = points)
	simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square = 4.0)

	print("0: ", end = '')
	print(*simplex_tree.get_skeleton(0))
	print("1: ", end = "")
	print(*simplex_tree.get_skeleton(1))
	print("2: ", end = "")
	print(*simplex_tree.get_skeleton(2))

	# Plot the points
	plt.scatter(points[:, 0], points[:, 1])

	# Plot the edges of the alpha complex
	for simplex in simplex_tree.get_skeleton(1):

		if len(simplex[0]) == 2:

			i, j = simplex[0]
			plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k-')

	plt.show()

	# Find the agent's voronoi neighbors
	# def ComputeNeighbors(self):

	# points = np.array([(1, 1), (2, 3), (4, 2), (5, 4), (8, 8), (8, 3)])
	# tri = Delaunay(points)

	# ids = []
	# for simplex in tri.simplices:

	# 	if idx_map[self.id] in simplex:

	# 		for id_ in simplex:

	# 			ids.append(id_)

	# neighbors = []
	# for member in self.neighbors.keys():

	# 	if idx_map[member] in ids:

	# 		neighbors.append(member)

	# return neighbors

def Kmeans():

	# Generate some example data
	# np.random.seed(42)
	# data = np.random.rand(100, 2)
	data = np.array([(1, 1), (2, 3), (4, 2), (5, 4), (8, 8), (8, 3)])

	# Specify the number of clusters (k)
	k = 5

	# Create a k-means model
	kmeans = KMeans(n_clusters=k)

	# Fit the model to the data
	kmeans.fit(data)

	# Get the cluster centroids
	centroids = kmeans.cluster_centers_

	print("centroids: " + str(centroids))

	# Get the labels assigned to each data point (which cluster it belongs to)
	labels = kmeans.labels_

	# Plot the data points and cluster centroids
	plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k')
	plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
	plt.legend()
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('K-Means Clustering')
	plt.show()

def Kmeansb():

	data = np.array([(1, 1), (2, 3), (4, 2), (5, 4)])
	centroids = np.array([(8,8)])
	alpha = 0.3

	for i in range(100):

		labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
		mean = np.array([data[labels == i].mean(axis=0) for i in range(len(centroids))])

		new_centroids = (1 - alpha)*centroids + alpha*mean

		if np.allclose(centroids, new_centroids):

			break
		else:

			centroids = new_centroids

		plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k')
		plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200, label='Centroids')
		plt.legend()
		plt.xlabel('Feature 1')
		plt.ylabel('Feature 2')
		plt.title('K-Means Clustering')
		plt.show()

	print("Mean: " + str(mean))
	print("Centroid: " + str(centroids))

def NN():

	# Example usage
	points = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0), (10.5, 12.5), 
						(12.5, 9.5), (16.5, 12.5), (12.5, 17.5)])
	num_points = len(points)
	visited = [False] * num_points
	path = []

	# Start from a random point
	current_point_idx = 0

	for _ in range(num_points):

		path.append(points[current_point_idx])
		visited[current_point_idx] = True

		# Find the nearest unvisited point
		min_distance = float('inf')
		nearest_point_idx = None

		for i in range(num_points):

			if not visited[i]:

				distance = np.linalg.norm(points[current_point_idx] - points[i])

				if distance < min_distance:

					min_distance = distance
					nearest_point_idx = i

		current_point_idx = nearest_point_idx

	print(path)

	# Extract x and y coordinates for plotting
	x = points[:, 0]
	y = points[:, 1]

	# Create the plot
	plt.figure(figsize=(8, 6))
	plt.scatter(x, y, c='blue', label='Points')
	plt.plot([point[0] for point in path], [point[1] for point in path], c='red', marker='o', linestyle='-', label='Hamiltonian Path')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Nearest Neighbor Algorithm for Hamiltonian Path')
	plt.legend()
	plt.grid(True)
	plt.show()

def distinguish_data():

	points = [[1, 2], [4, 5], [5, 7], [8, 9]]
	threshold = np.sqrt(2)

	continuous_groups = []
	current_group = []

	for i in range(len(points) - 1):

		# distance = np.linalg.norm(np.array(points[i]) - np.array(points[i + 1]))


		if points[i][1] == points[i+1][0]:

			current_group.append(points[i])
		else:

			current_group.append(points[i])
			continuous_groups.append(current_group)
			current_group = []

	# Append the last point to the current group
	current_group.append(points[-1])
	continuous_groups.append(current_group)

	independent_points = []

	for i in range(len(points)):

		is_independent = True

		for j in range(len(points)):

			if i != j:

				distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))

				if distance <= threshold:

					is_independent = False
					break

		if is_independent:

			independent_points.append(points[i])

	print("Continuous Groups:", continuous_groups)
	print("Independent Points:", independent_points)

def spc():

	# Generate example data
	np.random.seed(0)
	n_samples = 8
	# X = np.random.rand(n_samples, 2)
	X = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0),
				(10.5, 12.5), (12.5, 9.5), (16.5, 12.5), (12.5, 17.5)])

	# Create a SpectralClustering instance
	n_clusters = 4
	spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=8)

	# Fit and predict clusters
	cluster_labels = spectral_clustering.fit_predict(X)

	# Plot the clustering result
	plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='rainbow')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Spectral Clustering Result')
	plt.show()

def MS():

	# Generate example data
	np.random.seed(0)
	# n_samples = 10
	# X = np.random.rand(n_samples, 2)
	X = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0),
				(10.5, 12.5), (12.5, 9.5), (16.5, 12.5), (12.5, 17.5)])

	# Create a MeanShift instance
	bandwidth = 2
	mean_shift_clustering = MeanShift(bandwidth=bandwidth)

	# Fit and predict clusters
	cluster_labels = mean_shift_clustering.fit_predict(X)
	cluster_centers = mean_shift_clustering.cluster_centers_
	print("cluster labels: " + str(cluster_labels))
	print("cluster_centers: " + str(cluster_centers))

	# Plot the clustering result
	plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='rainbow')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Mean Shift Clustering Result')
	plt.show()

def HA_BGC():

	# Targets' and trackers' positions
	targets = np.array([[1, 2], [4, 5], [5, 6], [9, 11], [3, 8], [7, 10], [12, 4], [6, 9]])
	trackers = np.array([[2, 4], [7, 8], [11, 10], [5, 5]])

	# Combine targets and trackers for clustering
	data = np.vstack((targets, trackers))

	# Calculate distances between targets and trackers
	distances = np.linalg.norm(targets[:, np.newaxis, :] - trackers, axis=-1)

	# Convert distances to costs (negate)
	costs = -distances

	# Apply Hungarian Algorithm for assignment
	row_indices, col_indices = linear_sum_assignment(costs)

	# Assign targets to trackers based on assignment
	assignments = {tracker_idx: [] for tracker_idx in range(len(trackers))}

	for target_idx, tracker_idx in zip(row_indices, col_indices):

		assignments[tracker_idx].append(target_idx)

	# Print the assignments
	for tracker_idx, assigned_targets in assignments.items():

		print(f"Tracker {tracker_idx} assigned to targets: {assigned_targets}")

	# Visualize the clusters
	plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
	plt.scatter(trackers[:, 0], trackers[:, 1], marker='X', color='black', label='Trackers')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Normalized Cut Clustering Result')
	plt.legend()
	plt.show()

def NCA_GBC():

	# Targets' and trackers' positions
	targets = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0)])
	trackers = np.array([[2, 4], [7, 8], [11, 10], [5, 5]])

	# Combine targets and trackers for clustering
	data = np.vstack((targets, trackers))

	# Calculate affinity matrix (similarity)
	affinity_matrix = np.exp(-pairwise_distances(data, metric='euclidean'))
	print("affinity matrix: " + str(affinity_matrix))

	# Apply spectral clustering to obtain clusters
	num_clusters = len(trackers)
	clusters = spectral_clustering(affinity_matrix, n_clusters=num_clusters, eigen_solver='arpack')

	# Separate targets and trackers into clusters
	target_clusters = clusters[:len(targets)]
	tracker_clusters = clusters[len(targets):]

	# Print the assignments
	for i, (target, tracker) in enumerate(zip(targets, trackers)):

		print(f"Target {i} assigned to Tracker {tracker_clusters[i]}")
	
	# Visualize the clusters
	plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
	plt.scatter(trackers[:, 0], trackers[:, 1], marker='X', color='black', label='Trackers')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('Normalized Cut Clustering Result')
	plt.legend()
	plt.show()

def OPTICS_DBC():

	# Targets' and trackers' positions
	targets = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0),
				(10.5, 12.5), (12.5, 9.5), (16.5, 12.5), (12.5, 17.5)])
	trackers = np.array([[2, 2], [2, 23], [23, 23], [23, 2]])

	# Combine targets and trackers for clustering
	data = np.vstack((targets, trackers))

	# Apply OPTICS clustering
	min_samples = 2  # Minimum number of points to form a dense region
	optics = OPTICS(min_samples=min_samples)
	clusters = optics.fit_predict(data)

	# Separate targets and trackers into clusters
	target_clusters = clusters[:len(targets)]
	tracker_clusters = clusters[len(targets):]

	# Print the assignments
	for i, (target, tracker) in enumerate(zip(targets, trackers)):

		print(f"Target {i} assigned to Tracker {tracker_clusters[i]}")

	# Visualize the clusters
	plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='rainbow')
	plt.scatter(trackers[:, 0], trackers[:, 1], marker='X', color='black', label='Trackers')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('OPTICS Clustering Result')
	plt.legend()
	plt.show()

def Array_test():

	A = [[]]
	print(np.shape(A)[0])
	print(type(np.shape(A)[1]))
	data = [[(4, 0), (0, 1)], [(1, 3), (3, 2)], [(2, 5), (5, 9), (9, 11)], [(11, 15), (15, 19), (19, 31)]]

	# Find the element with maximum size
	max_element = max(data, key=len)
	print("Element with maximum size:", max_element)

	# Find elements with identical sizes
	size_count = {}
	size_count[len(max_element)] = [max_element]
	print("size_count: " + str(size_count))

	for element in data:

		size = len(element)

		if size == len(max_element) and not np.all(np.array(element) == np.array(size_count[len(max_element)])):

			size_count[size].append(element)

		# if size in size_count:

		# 	size_count[size].append(element)
		# else:

		# 	size_count[size] = [element]

	print("size_count: " + str(size_count))
	
	# Filter elements with identical sizes
	identical_size_elements = [elements for size, elements in size_count.items()][0]

	print("Elements with identical sizes:", identical_size_elements)

def random_test():

	seed_key = {1: "000", 2: "103", 3: "106"}

	speed_gain = 0.1

	for i in range(3):

		for j in range(1, len(seed_key)+1):

			np.random.seed(int(seed_key[j]))
			velocities = (float(speed_gain)/0.5)*(np.random.rand(8, 2) - 0.5)

			print("i, j, velocities: ", end = "")
			print(i, j, velocities)

def Hierarchical_Clustering():

	# Sample data points
	data = np.array([[1, 2], [2, 3], [3, 4], [6, 9], [10, 11], [12, 15]])
	# np.random.seed(0)
	# num_points = 30
	# data = np.random.rand(num_points, 2)

	# Perform Agglomerative Hierarchical Clustering using complete linkage
	linkage_matrix = linkage(data, method = 'complete')

	clusters_at_levels = {}
	for i in range(1, len(linkage_matrix) + 1):

		cluster_assignments = fcluster(linkage_matrix, i, criterion='maxclust')
		clusters_at_levels[i] = cluster_assignments

	# Print clusters at each level
	for level, clusters in enumerate(clusters_at_levels, start=1):

		print(f"Clusters at level {level}: {clusters}")

	# Print members in each cluster at different levels
	for level, clusters in clusters_at_levels.items():

		cluster_dict = {}

		for idx, cluster_id in enumerate(clusters, start=1):

			if cluster_id not in cluster_dict:

				cluster_dict[cluster_id] = []
				cluster_dict[cluster_id].append(idx)

		print(f"Members in clusters at level {level}: {cluster_dict}")

	# Create a dendrogram
	dendrogram(linkage_matrix)

	plt.xlabel('Data Points')
	plt.ylabel('Distance')
	plt.title('Agglomerative Hierarchical Clustering Dendrogram')
	plt.show()

	# Determine cluster memberships using a distance threshold
	distance_threshold = np.sqrt(2) # Set the distance threshold
	clusters = fcluster(linkage_matrix, t = distance_threshold, criterion = 'distance')

	print("Cluster memberships: ", clusters)

def HC_Step_2():

	# Sample data points
	data = np.array([[1, 2], [2, 3], [3, 4], [6, 9], [10, 11], [11, 12]])

	# Custom distance threshold for merging clusters
	threshold = np.sqrt(2.1)  # Adjust as needed

	# Initialize cluster assignments for each data point
	num_points = len(data)
	cluster_assignments = list(range(num_points))

	print("num_points: ", num_points)
	print("cluster_assignments: ", cluster_assignments)

	# Perform Agglomerative Hierarchical Clustering based on custom threshold
	for i in range(num_points):

		for j in range(i + 1, num_points):

			if euclidean(data[i], data[j]) < threshold:

				cluster_assignments[j] = cluster_assignments[i]

	print("cluster_assignments: ", cluster_assignments)
	# Get unique cluster IDs
	unique_clusters = set(cluster_assignments)
	print("unique_clusters: ", unique_clusters)

	# Assign cluster IDs to data points
	cluster_mapping = {cluster_id: [] for cluster_id in unique_clusters}
	print("cluster_mapping: ", cluster_mapping)

	for i, cluster_id in enumerate(cluster_assignments):

		cluster_mapping[cluster_id].append(i)

	print("cluster_mapping: ", cluster_mapping)

	# Print cluster assignments
	i = 0
	cluster = {}
	for cluster_id, points in cluster_mapping.items():

		cluster[i] = points
		i += 1
	
	print("Cluster: ", cluster)

def HC_Step_5():

	# Sample data points
	data_points = np.array([[1, 2], [2, 3], [4, 5], [9, 10], [10, 11]])

	# Custom distance threshold for merging clusters
	threshold = np.sqrt(4)  # Adjust as needed

	# Initialize each data point as its own cluster
	clusters = [[point] for point in data_points]
	cluster_mapping = {index_: [index_] for (index_, element) in enumerate(data_points)}
	cluster_mapping_save = {str(0): [] for (index_, element) in enumerate(data_points)}
	print("clusters: ", clusters)
	print("cluster_mapping: ", cluster_mapping, "\n")

	# Loop until only one cluster remains
	count = 0

	while set(cluster_mapping.keys()) != set(cluster_mapping_save.keys()) or count > 100:

		cluster_mapping_save = cluster_mapping

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

				if dist < min_distance and dist < threshold:

					min_distance = dist
					min_i, min_j = i, j

		if min_i != -1 and min_j != -1:

			print("min_i, min_j: ", min_i, min_j)
		
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

			print("min_i, min_j: ", min_i, min_j)

		print("clusters: ", clusters)
		print("cluster_mapping: ", cluster_mapping, "\n")

		count += 1

	print("clusters: ", clusters)
	print("cluster_mapping: ", cluster_mapping)

def One_hop_neighbor():

	# Generate random points
	np.random.seed(0)
	points = np.array([[2.0, 2.0], [23.0, 2.0], [2.0, 23.0], [23.0, 23.0]])

	# Create Delaunay Triangulation
	tri = Delaunay(points)

	# Find one-hop neighbors for each point
	one_hop_neighbors = [[] for _ in range(len(points))]

	print("one_hop_neighbors: ", one_hop_neighbors)

	for simplex in tri.simplices:

		print("simplex: ", simplex)

		for point_index in simplex:

			for neighbor_index in simplex:

				if point_index != neighbor_index and neighbor_index not in one_hop_neighbors[point_index]:

					one_hop_neighbors[point_index].append(neighbor_index)

	print("one_hop_neighbors: ", one_hop_neighbors)

	# Print the one-hop neighbors of each point
	for i, neighbors in enumerate(one_hop_neighbors):

		print(f"Point {i} is connected to: {neighbors}")

	# Plot the Delaunay Triangulation
	plt.triplot(points[:, 0], points[:, 1], tri.simplices)
	plt.plot(points[:, 0], points[:, 1], 'o')
	plt.title('Delaunay Triangulation')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

def Tidal_Locking():

	# Constants
	center = np.array([0, 0])  # Center point with an offset
	rotating_radius = 5  # Distance between rotating point and center
	angular_velocity = 0.1  # Angular velocity (radians per time step)
	time_steps = 500  # Number of time steps
	time = np.linspace(0, 2*np.pi, time_steps)  # Time values

	# Initialize arrays to store positions and orientations
	rotating_positions = np.zeros((time_steps, 2))
	rotating_orientations = np.zeros(time_steps)

	# Simulate motion
	for i, angle in enumerate(time):

		position = center + rotating_radius * np.array([np.cos(angle), np.sin(angle)])
		orientation = np.arctan2(center[1] - position[1], center[0] - position[0])
		rotating_positions[i] = position
		rotating_orientations[i] = orientation

	# Plot the results
	plt.figure(figsize=(8, 6))
	plt.plot(rotating_positions[:, 0], rotating_positions[:, 1], label="Rotating Point")
	plt.plot(center[0], center[1], "ro", label="Center Point")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("Rotating Point Simulation with Offset Center")
	plt.legend()
	plt.axis("equal")
	plt.grid()
	plt.show()

def FuzzyCMeans():

	# Step 1: Generate sample data (you can replace this with your data)
	n_samples = 300
	n_features = 2
	n_clusters = 3
	data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

	# Step 2: Normalize the data (optional but recommended)
	data = (data - data.min()) / (data.max() - data.min())

	# Step 3: Configure and run fuzzy c-means clustering
	n_clusters = 3  # Number of clusters
	fuzziness = 2.0  # Fuzziness parameter (usually >= 1)
	error_threshold = 0.005  # Stop criterion for convergence
	max_iterations = 1000  # Maximum number of iterations

	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
	data.T, n_clusters, fuzziness, error_threshold, max_iterations, seed=42
	)

	# Step 4: Analyze the results
	cluster_membership = np.argmax(u, axis=0)  # Get the cluster membership for each point

	# Visualize the clusters
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

	for i in range(n_clusters):

		plt.scatter(data[cluster_membership == i, 0], data[cluster_membership == i, 1], c=colors[i], label=f'Cluster {i+1}')

	plt.scatter(cntr[:, 0], cntr[:, 1], marker='*', s=200, c='k', label='Cluster centers')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.legend()
	plt.title('Fuzzy C-Means Clustering')
	plt.show()


if __name__ == '__main__':

	# MST2MSF()
	# print("----------------------------")
	# SEMST()

	# HA()
	# concavehull()
	# alpha_complex()
	# Kmeansb()
	# NN()
	# distinguish_data()
	# spc()
	# MS()
	# HA_BGC()
	# NCA_GBC()
	# OPTICS_DBC()
	# random_test()
	# Hierarchical_Clustering()
	# HC_Step_2()
	# HC_Step_5()
	# One_hop_neighbor()
	# Tidal_Locking()
	# FuzzyCMeans()

	sequence_num = [1,2,3,4,5]
	row_ind = [1,2,3,4,5]
	missing_numbers = [num for num in sequence_num if num not in row_ind]
	print(len(missing_numbers))