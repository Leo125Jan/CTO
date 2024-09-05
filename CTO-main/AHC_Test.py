import numpy as np

def Dendrogram_Construction(Data):

	dendrogram = []

	for point in Data:

		m = {"point": None, "left": None, "right": None, "distance": None}
		m["point"] = point
		m["left"] = None
		m["right"] = None
		m["distance"] = 0.0

		dendrogram.append(m)

	print("dendrogram: ", dendrogram)

	for i in range(len(dendrogram)-1):

		print("\n")
		print("Interion: ", i)

		pair, min_distance = FindClosestPair(dendrogram)
		Merge(pair, min_distance, dendrogram)

	# print(halt)

	return dendrogram[0]

def FindClosestPair(dendrogram):

	min_distance = np.inf
	closest_pair = None

	for i in range(len(dendrogram)-1):

		for j in range(i+1, len(dendrogram)):

			distance = LinkageDistance(dendrogram[i], dendrogram[j])

			if distance < min_distance:

				min_distance = distance
				closest_pair = (i, j)

	# print("min_distance: ", min_distance)
	# print("closest_pair: ", closest_pair)
	# print(halt)

	return closest_pair, min_distance

def LinkageDistance(node1, node2):

	# print("node1: ", node1)
	# print("node2: ", node2)
	# print(halt)

	cluster1, cluster2 = [], []
	# cluster1 = Inter_Dendrogram_Walk(cluster1, node1)
	Inter_Dendrogram_Walk(cluster1, node1)
	print("cluster1: ", cluster1)
	# cluster2 = Inter_Dendrogram_Walk(cluster2, node2)
	Inter_Dendrogram_Walk(cluster2, node2)
	print("cluster2: ", cluster2)
	# print(halt)
	centroid1 = np.mean(cluster1, axis=0)
	centroid2 = np.mean(cluster2, axis=0)
	# print("centroid1: ", centroid1)
	# print("centroid2: ", centroid2)
	# print(halt)

	distance = np.linalg.norm(centroid1-centroid2)
	# print("distance: ", distance)
	# print(halt)

	return distance

def Merge(pair, min_distance, dendrogram):

	z = {"point": None, "left": None, "right": None, "distance": None}
	z["point"] = None
	z["left"] = dendrogram[pair[0]]
	z["right"] = dendrogram[pair[1]]
	z["distance"] = min_distance
	print("z: ", z)
	# print(halt)

	hold = [dendrogram[i] for i in range(len(dendrogram)) if i not in pair]
	dendrogram[:] = hold
	# print("dendrogram: ", dendrogram)
	dendrogram.append(z)
	# print("dendrogram: ", dendrogram)
	# print(halt)

def Inter_Dendrogram_Walk(cluster, node):

	if node != None:

		Inter_Dendrogram_Walk(cluster, node["left"])

		if node["distance"] == 0 and (node["point"] != None).all():

			cluster.append(node["point"])

		Inter_Dendrogram_Walk(cluster, node["right"])

	# print("cluster: ", cluster)

	# return cluster

def dendrogramCut(Cluster, node, distance_threshold):

	if node["distance"] <= distance_threshold:

		cluster = []
		Inter_Dendrogram_Walk(cluster, node)
		Cluster.append(cluster)
	else:

		dendrogramCut(Cluster, node["left"], distance_threshold)
		dendrogramCut(Cluster, node["right"], distance_threshold)

def add_element(my_list):

	my_list.append(42)

if __name__ == "__main__":

	# points = np.array([(1,1), (2,2), (5,7)])
	points = np.array([(12.0, 12.0), (12.0, 13.0), (13.0, 12.0), (13.0, 13.0), (10.5, 12.5), 
				(12.5, 9.5), (16.5, 12.5), (12.5, 17.5)])

	dendrogram = Dendrogram_Construction(points)
	print("dendrogram: ", dendrogram)
	Cluster = []
	dendrogramCut(Cluster, dendrogram, 4.0)
	print("Cluster: ", Cluster)

	# original_list = [1, 2, 3]
	# print("Original list before function call:", original_list)

	# add_element(original_list)

	# print("Original list after function call:", original_list)