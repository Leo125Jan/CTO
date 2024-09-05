import time
import hdbscan
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# if __name__ == '__main__':

# 	# Let's assume you have some data points
# 	data = np.array([[1, 2], [1, 4], [1, 0]])

# 	# Define initial centers
# 	init_centers = np.array([[1, 0], [2, 2], [3, 4]])

# 	# Create a KMeans instance with 3 clusters and defined initial centers
# 	kmeans = KMeans(n_clusters=3, init=init_centers, n_init=1)

# 	# Fit the model to your data
# 	kmeans.fit(data)

# 	# Now you can get the cluster centers
# 	centers = kmeans.cluster_centers_

# 	# And predict the cluster index for each sample
# 	labels = kmeans.predict(data)

# 	print("Cluster centers:\n", centers)
# 	print("Labels:\n", labels)

# if __name__ == '__main__':

# 	# Data
# 	categories = ['Category 1', 'Category 2', 'Category 3']
# 	part1 = [5, 7, 8]
# 	part2 = [3, 2, 4]
# 	part3 = [2, 3, 5]

# 	# Create the figure and axes
# 	fig, ax = plt.subplots()

# 	# Position of the bars on the x-axis
# 	bar_width = 0.5
# 	bar_positions = np.arange(len(categories))

# 	# Plotting each part of the bars
# 	ax.bar(bar_positions, part1, bar_width, label='Part 1', color='b')
# 	ax.bar(bar_positions, part2, bar_width, bottom=part1, label='Part 2', color='r')
# 	ax.bar(bar_positions, part3, bar_width, bottom=np.array(part1) + np.array(part2), label='Part 3', color='g')

# 	# Adding labels and title
# 	ax.set_xlabel('Categories')
# 	ax.set_ylabel('Values')
# 	ax.set_title('Stacked Bar Chart Example')
# 	ax.set_xticks(bar_positions)
# 	ax.set_xticklabels(categories)
# 	ax.legend()

# 	# Show the plot
# 	plt.show()

# if __name__ == '__main__':

# 	# Define the data
# 	X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# 	# Compute DBSCAN
# 	db = DBSCAN(eps=1, min_samples=2).fit(X)  # Adjust the eps and min_samples as needed
# 	labels = db.labels_
# 	print("labels; ", labels)

# 	# Number of clusters in labels, ignoring noise if present
# 	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# 	n_noise_ = list(labels).count(-1)

# 	print('Estimated number of clusters: %d' % n_clusters_)
# 	print('Estimated number of noise points: %d' % n_noise_)

# 	# Black removed and is used for noise instead
# 	unique_labels = set(labels)
# 	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

# 	for k, col in zip(unique_labels, colors):

# 		if k == -1:

# 			# Black used for noise
# 			col = [0, 0, 0, 1]

# 		class_member_mask = (labels == k)

# 		xy = X[class_member_mask & (labels != -1)]
# 		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
# 									markeredgecolor='k', markersize=6)

# 		xy = X[class_member_mask & (labels == -1)]
# 		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
# 									markeredgecolor='k', markersize=6)

# 	plt.title('Estimated number of clusters: %d' % n_clusters_)
# 	plt.show()

# if __name__ == '__main__':

# 	# Define the matrix
# 	matrix = np.array([[1, 1, 0, 0, 0],
# 						[1, 0, 0, 1, 0],
# 						[0, 0, 0, 0, 0],
# 						[1, 1, 1, 1, 0],
# 						[0, 0, 1, 0, 0]])

# 	# Perform logical AND on each column
# 	column_and = np.logical_or.reduce(matrix, axis=0)

# 	# Converting boolean array to integer for display
# 	column_and = column_and.astype(int)

# 	print(column_and)
# 	print(np.sum(column_and))

# 	# Calculate the number of `1`s in each column
# 	count_ones = np.sum(matrix == 1, axis=0)
# 	print("count_ones: ", count_ones)

# 	# Calculate the amount of `1`s, `2`s, and the number of elements >= `3`
# 	num_ones = np.sum(count_ones >= 1)
# 	num_twos = np.sum(count_ones >= 2)
# 	num_threes_or_more = np.sum(count_ones >= 3)

# 	print(f"Number of 1s: {num_ones}")
# 	print(f"Number of 2s: {num_twos}")
# 	print(f"Number of elements >= 3: {num_threes_or_more}")

# if __name__ == '__main__':

# 	# Given array
# 	arr = np.array([4., 1., 2., 3., 1., 5., 6., 2.])

# 	# Count occurrences of each element in the array
# 	count = Counter(arr)

# 	# Number of unique elements
# 	num_unique_elements = len(count)

# 	# Total length of the array
# 	total_length = len(arr)

# 	# Calculate the number of identical elements
# 	num_identical_elements = total_length - num_unique_elements

# 	# print(f"Total number of elements: {total_length}")
# 	# print(f"Number of unique elements: {num_unique_elements}")
# 	# print(f"Number of identical elements: {num_identical_elements}")

# 	cluster_set = {0: np.array([0]), 1: np.array([1]), 2: np.array([2, 5]), 3: np.array([3]), 
# 					4: np.array([4]), 5: np.array([6]), 6: np.array([7])}

# 	# Find keys with value length greater than 1
# 	keys_with_large_values = [key for key, value in cluster_set.items() if len(value) > 1]

# 	print(keys_with_large_values)

# if __name__ == '__main__':

# 	arr = np.array([4., 1., 2., 3., 1., 5., 6., 2.])

# 	# Step 1: Identify identical elements
# 	unique, counts = np.unique(arr, return_counts=True)
# 	print("unique: ", unique)
# 	print("counts: ", counts)
# 	identical_elements = unique[counts > 1]

# 	# Step 2: Find indices of distinct elements
# 	distinct_elements = unique[counts == 1]
# 	distinct_indices = [i for i, x in enumerate(arr) if x in distinct_elements]

# 	print("Identical elements:", identical_elements)
# 	print("Indices of distinct elements:", distinct_indices)

# if __name__ == '__main__':

# 	Agent = np.array([[1,1], [2,2], [3,3], [4,4]])
# 	Target = np.array([[20.67, 8.37], [16.99, 7.56], [1, 1]])

# 	# Calculate the distance matrix
# 	distance_matrix = np.linalg.norm(Agent[:, np.newaxis] - Target, axis=2)

# 	# Print the distance matrix
# 	print("distance matrix: \n", distance_matrix)

# 	a = "0.90"
# 	print(str(int(float(a)*100)))

# def kmeans(X, k, max_iters, tol):

# 	n_samples, n_features = X.shape
# 	centroids = X[np.random.choice(n_samples, k, replace=False)]
# 	# centroids = np.ones((k, n_features))

# 	for _ in range(max_iters):

# 		distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
# 		clusters = np.argmin(distances, axis=1)
# 		new_centroids = np.array([X[clusters == j].mean(axis=0) for j in range(k)])

# 		if np.linalg.norm(new_centroids - centroids) < tol:

# 			break

# 		centroids = new_centroids

# 	return centroids, clusters

# def fuzzy_cmeans(X, c, m, max_iters, tol):

# 	n_samples, n_features = X.shape
# 	# print("n_samples, n_features: ", n_samples, n_features)
# 	U = np.random.dirichlet(np.ones(c), size=n_samples)
# 	# centroids = np.zeros((c, n_features))
# 	centroids = X[np.random.choice(n_samples, c, replace=False)]
# 	# print("U: \n", U)
# 	# print("centroids: \n", centroids)

# 	for _ in range(max_iters):

# 		U_m = U ** m
# 		centroids = (U_m.T @ X) / U_m.sum(axis=0)[:, None]
# 		distances = np.linalg.norm(X[:, None] - centroids, axis=2)
# 		distances = np.fmax(distances, np.finfo(np.float64).eps)
# 		new_U = 1 / distances
# 		new_U = new_U / new_U.sum(axis=1, keepdims=True)

# 		if np.linalg.norm(new_U - U) < tol:

# 			break

# 		U = new_U

# 	return centroids, U

# def agglomerative_clustering(X, n_clusters):

# 	n_samples = X.shape[0]
# 	# print("n_samples: ", n_samples)
# 	clusters = {i: [i] for i in range(n_samples)}
# 	# print("clusters: ", clusters)
# 	distances = np.linalg.norm(X[:, None] - X, axis=2)
# 	# print("distances: \n", distances)
# 	np.fill_diagonal(distances, np.inf)
# 	# print("distances: \n", distances)

# 	while len(clusters) > n_clusters:

# 		i, j = np.unravel_index(np.argmin(distances), distances.shape)
# 		# print("clusters: ", clusters)
# 		clusters[i].extend(clusters[j])
# 		# print("clusters: ", clusters)
# 		del clusters[j]
# 		# print("clusters: ", clusters)

# 		for k in range(n_samples):

# 			if k in clusters and k != i:

# 				distances[i, k] = distances[k, i] = min(distances[i, k], distances[j, k])

# 			distances[:, j] = distances[j, :] = np.inf

# 		# print("clusters: ", clusters)

# 	return clusters

# def dbscan(X, eps, min_samples):

# 	n_samples = X.shape[0]
# 	labels = -np.ones(n_samples)
# 	cluster_id = 0

# 	def region_query(point_idx):

# 		distances = np.linalg.norm(X - X[point_idx], axis=1)

# 		return np.where(distances <= eps)[0]

# 	def expand_cluster(point_idx, neighbors):

# 		labels[point_idx] = cluster_id
# 		i = 0

# 		while i < len(neighbors):

# 			neighbor_idx = neighbors[i]

# 			if labels[neighbor_idx] == -1:

# 				labels[neighbor_idx] = cluster_id
# 			elif labels[neighbor_idx] == 0:

# 				labels[neighbor_idx] = cluster_id
# 				new_neighbors = region_query(neighbor_idx)

# 				if len(new_neighbors) >= min_samples:

# 					neighbors = np.append(neighbors, new_neighbors)

# 			i += 1

# 	for point_idx in range(n_samples):

# 		if labels[point_idx] == -1:

# 			neighbors = region_query(point_idx)

# 		if len(neighbors) >= min_samples:

# 			cluster_id += 1
# 			expand_cluster(point_idx, neighbors)
# 		else:

# 			labels[point_idx] = 0

# 	return labels

# def hdbscan_(X, min_cluster_size):

# 	# Create an HDBSCAN instance
# 	clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)

# 	# Fit the model to the data
# 	clusterer.fit(X)

# 	# Get the labels (note that -1 indicates noise)
# 	labels = clusterer.labels_

# 	return labels

# if __name__ == '__main__':

# 	# Generate sample data
# 	n_samples = 40
# 	n_features = 2
# 	centers = 40

# 	X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=42)

# 	# Parameters for fair comparison
# 	max_iter = 1000
# 	tolerance = 1e-4

# 	# K-means Clustering
# 	start_time = time.time()
# 	# kmeans = KMeans(n_clusters=centers, max_iter=max_iter, tol=tolerance, random_state=42)
# 	# kmeans.fit(X)
# 	centroids_kmeans, clusters_kmeans = kmeans(X, centers, max_iter, tolerance)
# 	# print("centroids_kmeans: ", centroids_kmeans)
# 	kmeans_time = time.time() - start_time

# 	# Fuzzy C-means Clustering
# 	start_time = time.time()
# 	# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, centers, 2, error=tolerance, maxiter=max_iter, init=None)
# 	centroids_fcm, U_fcm = fuzzy_cmeans(X, centers, 2, max_iter, tolerance)
# 	# print("centroids_fcm: ", centroids_fcm)
# 	fuzzy_time = time.time() - start_time

# 	# DBSCAN Clustering
# 	start_time = time.time()
# 	# dbscan = DBSCAN(eps=0.5, min_samples=2)
# 	# dbscan.fit(X)
# 	labels_dbscan = dbscan(X, 0.5, 2)
# 	# print("labels_dbscan: ", labels_dbscan)
# 	dbscan_time = time.time() - start_time

# 	# Hierarchical Clustering
# 	start_time = time.time()
# 	# hierarchical = AgglomerativeClustering(n_clusters=centers)
# 	# hierarchical.fit(X)
# 	clusters_agg = agglomerative_clustering(X, centers)
# 	# print("clusters_agg: ", clusters_agg)
# 	hierarchical_time = time.time() - start_time

# 	# HDBSCAN
# 	start_time = time.time()
# 	clusters_labels = hdbscan_(X, 2)
# 	hdbscan_time = time.time() - start_time

# 	# Print results
# 	print("K-means time: ", kmeans_time)
# 	print("Fuzzy C-means time: ", fuzzy_time)
# 	print("DBSCAN time: ", dbscan_time)
# 	print("Hierarchical Clustering time: ", hierarchical_time)
# 	print("HDBSCAN time: ", hdbscan_time)

# if __name__ == '__main__':

# 	cluster_set = {0: np.array([0]), 1: np.array([1]), 2: np.array([2,5]), 3: np.array([3]), 4: np.array([4]),
# 					5: np.array([5]), 6: np.array([6])}
# 	agent_to_cluster = np.array([2,1,0,3,4,5,6,2])
# 	agents_position = np.array([[6.2, 18], [13.5, 3.9], [4.3, 14.65], [21.3, 14.5],
# 								[8, 6.5], [16.7, 12.2], [9.4, 21.8], [20, 20]])
# 	print("cluster_set: ", cluster_set)
# 	print("agent_to_cluster: ", agent_to_cluster)

# 	# Generate teammates
# 	teammates = []
# 	for i in range(len(agent_to_cluster)):

# 		cluster_id = agent_to_cluster[i]
# 		teammates.append(list(np.where(agent_to_cluster == cluster_id)[0]))
# 	print("teammates: ", teammates)

# 	# Convert teammates to unique sets
# 	unique_teammates = []
# 	for team in teammates:

# 		if team not in unique_teammates:

# 			unique_teammates.append(team)
# 	print("unique_teammates: ", unique_teammates)

# 	a = [1,2]
# 	a.remove(1)
# 	print(a)

if __name__ == '__main__':

	# Create some data
	x = np.linspace(0, 2*np.pi, 100)
	y = np.sin(x)

	fig, ax1 = plt.subplots()

	# Plot the data on the left y-axis
	ax1.plot(x, y, 'g-')
	ax1.set_ylabel('Y')
	ax1.tick_params('y')

	# Create a second y-axis that shares the same x-axis
	ax2 = ax1.twinx()

	# Copy the y-limits from the first axis
	ax2.set_ylim(ax1.get_ylim())
	# ax2.set_yticklabels([])  # Hide the tick labels

	plt.show()