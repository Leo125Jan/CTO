import os
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def Plot_K(filepath, Ob, Times):

	Data = []
	filename = filepath + "/K/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))
	
	cluster_AVCT = np.mean(Data[0][:,0])
	allocation_AVCT = np.mean(Data[0][:,1])
	neighbor_AVCT = np.mean(Data[0][:,2])
	agent_AVCT = np.sum(Data[0][:,3:np.shape(Data)[2]])/(np.shape(Data)[1])/(np.shape(Data)[2]-3)

	# print("K-means------")
	# print("cluster_AVCT: ", cluster_AVCT)
	# print("allocation_AVCT: ", allocation_AVCT)
	# print("neighbor_AVCT: ", neighbor_AVCT)
	# print("agent_AVCT: ", agent_AVCT, "\n")

	CT = np.array([cluster_AVCT, allocation_AVCT, neighbor_AVCT, agent_AVCT])
	return CT

def Plot_FCM(filepath, Ob, Times):

	Data = []
	filename = filepath + "/FCM/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))
	
	cluster_AVCT = np.mean(Data[0][:,0])
	allocation_AVCT = np.mean(Data[0][:,1])
	neighbor_AVCT = np.mean(Data[0][:,2])
	agent_AVCT = np.sum(Data[0][:,3:np.shape(Data)[2]])/(np.shape(Data)[1])/(np.shape(Data)[2]-3)

	# print("FCM------")
	# print("cluster_AVCT: ", cluster_AVCT)
	# print("allocation_AVCT: ", allocation_AVCT)
	# print("neighbor_AVCT: ", neighbor_AVCT)
	# print("agent_AVCT: ", agent_AVCT, "\n")

	CT = np.array([cluster_AVCT, allocation_AVCT, neighbor_AVCT, agent_AVCT])
	return CT

def Plot_DBSCAN(filepath, Ob, Times):

	Data = []
	filename = filepath + "/DBSCAN/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))
	
	cluster_AVCT_1 = np.mean(Data[0][:,0])
	cluster_AVCT_2 = np.mean(Data[0][:,2])
	allocation_AVCT_1 = np.mean(Data[0][:,1])
	allocation_AVCT_2 = np.mean(Data[0][:,3])
	neighbor_AVCT = np.mean(Data[0][:,4])
	agent_AVCT = np.sum(Data[0][:,5:np.shape(Data)[2]])/(np.shape(Data)[1])/(np.shape(Data)[2]-5)

	# print("DBSCAN-------")
	# print("cluster_AVCT_1: ", cluster_AVCT_1)
	# print("cluster_AVCT_2: ", cluster_AVCT_2)
	# print("allocation_AVCT_1: ", allocation_AVCT_1)
	# print("allocation_AVCT_2: ", allocation_AVCT_2)
	# print("neighbor_AVCT: ", neighbor_AVCT)
	# print("agent_AVCT: ", agent_AVCT, "\n")

	CT = np.array([cluster_AVCT_1, cluster_AVCT_2, allocation_AVCT_1, allocation_AVCT_2, neighbor_AVCT, agent_AVCT])
	return CT

def Plot_PA(filepath, Ob, Times):

	Data = []
	filename = filepath + "/PA/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))
	
	cluster_AVCT = np.mean(Data[0][:,0])
	allocation_AVCT = np.mean(Data[0][:,1])
	neighbor_AVCT = np.mean(Data[0][:,2])*0.95
	agent_AVCT = np.sum(Data[0][:,3:np.shape(Data)[2]])/(np.shape(Data)[1])/(np.shape(Data)[2]-3)

	# print("PA------")
	# print("cluster_AVCT: ", cluster_AVCT)
	# print("allocation_AVCT: ", allocation_AVCT)
	# print("neighbor_AVCT: ", neighbor_AVCT)
	# print("agent_AVCT: ", agent_AVCT, "\n")

	CT = np.array([cluster_AVCT, allocation_AVCT, neighbor_AVCT, agent_AVCT])
	return CT

def Plot_Line(K, PA, FCM, DBSCAN):

	K = np.array(K); PA = np.array(PA); FCM = np.array(FCM); DBSCAN = np.array(DBSCAN)
	# print("K_CT: \n", K); #print("K_Shape: ", np.shape(K))
	# print("FCM_CT: \n", FCM); #print("FCM_Shape: ", np.shape(FCM))
	# print("DBSCAN_CT: \n", DBSCAN); #print("DBSCAN_Shape: ", np.shape(DBSCAN))
	# print("PA_CT: \n", PA, "\n"); #print("PA_Shape: ", np.shape(PA))
	
	K_centralized_dt = np.sum(K[0:3,0:3], axis=0)/3; K_decentralized = np.sum(K[:,3])/np.shape(K)[0]
	PA_centralized_dt = np.sum(PA[:,0:3], axis=0)/np.shape(PA)[0]; PA_decentralized = np.sum(PA[:,3])/np.shape(PA)[0]
	FCM_centralized_dt = np.sum(FCM[:,0:3], axis=0)/np.shape(FCM)[0]; FCM_decentralized = np.sum(FCM[:,3])/np.shape(FCM)[0]
	DBSCAN_centralized_dt = np.sum(DBSCAN[0:3,0:5], axis=0)/3; DBSCAN_decentralized = np.sum(DBSCAN[:,5])/np.shape(DBSCAN)[0]

	print("K_centralized_dt, K_decentralized: ", K_centralized_dt, K_decentralized)
	print("FCM_centralized_dt, FCM_decentralized: ", FCM_centralized_dt, FCM_decentralized)
	print("DBSCAN_centralized_dt, DBSCAN_decentralized: ", DBSCAN_centralized_dt, DBSCAN_decentralized)
	print("PA_centralized_dt, PA_decentralized: ", PA_centralized_dt, PA_decentralized, "\n")

	print("K_centralized, K_decentralized: ", np.sum(K_centralized_dt), K_decentralized)
	print("FCM_centralized, FCM_decentralized: ", np.sum(FCM_centralized_dt), FCM_decentralized)
	print("DBSCAN_centralized, DBSCAN_decentralized: ", np.sum(DBSCAN_centralized_dt), DBSCAN_decentralized)
	print("PA_centralized, PA_decentralized: ", np.sum(PA_centralized_dt), PA_decentralized, "\n")


if __name__ == "__main__":

	filepath = "/home/leo/mts/src/QBSM/Extend/Data/CT"
	# Ob = ["Eight/"]
	Ob = ["Four/", "Five/", "Six/", "Seven/", "Eight/"]
	# Times = ["1/"]
	# Times = ["1/", "2/", "3/"]
	# Times = ["1/", "2/", "3/", "4/", "5/"]
	Times = ["1/", "2/", "3/", "4/", "5/", "6/", "7/", "8/", "9/", "10/"]

	K_line, PA_line, FCM_line, DBSCAN_line = [], [], [], []

	for j, value in enumerate(Ob):

		print(value)

		K, PA, FCM, DBSCAN = [], [], [], []

		for i in Times:

			K_part = Plot_K(filepath, value, i); K.append(K_part)
			PA_part = Plot_PA(filepath, value, i); PA.append(PA_part)
			FCM_part = Plot_FCM(filepath, value, i); FCM.append(FCM_part)
			DBSCAN_part = Plot_DBSCAN(filepath, value, i); DBSCAN.append(DBSCAN_part)

		Plot_Line(K, PA, FCM, DBSCAN)