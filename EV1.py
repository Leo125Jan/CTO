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
from scipy.optimize import linear_sum_assignment

from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from collections import defaultdict

a = np.arange(0, 30, 0.1)
b = np.arange(0, 30, 0.1)
X, Y = np.meshgrid(a, b)

W = np.vstack([X.ravel(), Y.ravel()])
W = W.transpose()

def hura():

	cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
	row_ind, col_ind = linear_sum_assignment(cost)

	print("row: " + str(row_ind))
	print("col: " + str(col_ind))

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


if __name__ == '__main__':

	center_x, center_y, radius = circumcenter(1,0,2,4,3,0)
	print(center_x, center_y, radius)

	# MST2MSF()
	# print("----------------------------")
	# SEMST()

	a = np.array([2,2])
	b = np.array([2,3])
	c = ne.evaluate("a*b")

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