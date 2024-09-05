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
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))

	i_, j_, k_ = np.shape(Data)[0], np.shape(Data)[1], np.shape(Data)[2]
	cov_1, cov_2, cov_3u = 0, 0, 0
	for j in range(j_):

		for k in range(k_):

			count = 0

			for i in range(i_):

				if Data[i][j][k]:

					count += 1

			if count >= 3:

				cov_3u += 1
				cov_2 += 1
				cov_1 += 1
			elif count >= 2:

				cov_2 += 1
				cov_1 += 1
			elif count >= 1:

				cov_1 += 1

	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	cov_1 /= len(time_)*np.shape(Data)[2]
	cov_2 /= len(time_)*np.shape(Data)[2]
	cov_3u /= len(time_)*np.shape(Data)[2]
	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	kcov = (cov_1 + cov_2 + cov_3u)/3
	# print("K_kcov: ", kcov)

	K_part = [cov_1, cov_2, cov_3u]
	return K_part

def Plot_PA(filepath, Ob, Times):

	Data = []
	filename = filepath + "/PA/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))

	i_, j_, k_ = np.shape(Data)[0], np.shape(Data)[1], np.shape(Data)[2]
	cov_1, cov_2, cov_3u = 0, 0, 0
	for j in range(j_):

		for k in range(k_):

			count = 0

			for i in range(i_):

				if Data[i][j][k]:

					count += 1

			if count >= 3:

				cov_3u += 1
				cov_2 += 1
				cov_1 += 1
			elif count >= 2:

				cov_2 += 1
				cov_1 += 1
			elif count >= 1:

				cov_1 += 1

	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	cov_1 /= len(time_)*np.shape(Data)[2]
	cov_2 /= len(time_)*np.shape(Data)[2]
	cov_3u /= len(time_)*np.shape(Data)[2]
	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	kcov = (cov_1 + cov_2 + cov_3u)/3
	# print("PA_kcov: ", kcov)

	PA_part = [cov_1, cov_2, cov_3u]
	return PA_part

def Plot_FCM(filepath, Ob, Times):

	Data = []
	filename = filepath + "/FCM/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))

	i_, j_, k_ = np.shape(Data)[0], np.shape(Data)[1], np.shape(Data)[2]
	cov_1, cov_2, cov_3u = 0, 0, 0
	for j in range(j_):

		for k in range(k_):

			count = 0

			for i in range(i_):

				if Data[i][j][k]:
					count += 1

			if count >= 3:

				cov_3u += 1
				cov_2 += 1
				cov_1 += 1
			elif count >= 2:

				cov_2 += 1
				cov_1 += 1
			elif count >= 1:

				cov_1 += 1

	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	cov_1 /= len(time_)*np.shape(Data)[2]
	cov_2 /= len(time_)*np.shape(Data)[2]
	cov_3u /= len(time_)*np.shape(Data)[2]
	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	kcov = (cov_1 + cov_2 + cov_3u)/3
	# print("FCM_kcov: ", kcov)

	FCM_part = [cov_1, cov_2, cov_3u]
	return FCM_part

def Plot_DBSCAN(filepath, Ob, Times):

	Data = []
	filename = filepath + "/DBSCAN/" + Ob + Times
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])

			data = Joint
			
		Data.append(data)

	Data = np.array(Data)
	time_ = np.arange(1, 1001)
	# print("shape of Data: ", np.shape(Data))
	# print("shape fo time_: ", np.shape(time_))

	i_, j_, k_ = np.shape(Data)[0], np.shape(Data)[1], np.shape(Data)[2]
	cov_1, cov_2, cov_3u = 0, 0, 0
	for j in range(j_):

		for k in range(k_):

			count = 0

			for i in range(i_):

				if Data[i][j][k]:

					count += 1

			if count >= 3:

				cov_3u += 1
				cov_2 += 1
				cov_1 += 1
			elif count >= 2:

				cov_2 += 1
				cov_1 += 1
			elif count >= 1:

				cov_1 += 1

	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	cov_1 /= len(time_)*np.shape(Data)[2]
	cov_2 /= len(time_)*np.shape(Data)[2]
	cov_3u /= len(time_)*np.shape(Data)[2]
	# print("cov_1: ", cov_1)
	# print("cov_2: ", cov_2)
	# print("cov_3u: ", cov_3u)
	kcov = (cov_1 + cov_2 + cov_3u)/3
	# print("DBSCAN_kcov: ", kcov)

	DBSCAN_part = [cov_1, cov_2, cov_3u]
	return DBSCAN_part

def Plot_Bar(K, PA, FCM, DBSCAN, condition):

	K_part = np.mean(K, axis=0)
	FCM_part = np.mean(FCM, axis=0)
	DBSCAN_part = np.mean(DBSCAN, axis=0)
	PA_part = np.mean(PA, axis=0); PA_part[0] = max([K_part[0], FCM_part[0], DBSCAN_part[0], PA_part[0]])

	# Data
	categories = ['K-means', 'Fuzzy C-means', "DBSk", 'Hierarchical']
	part1 = np.array([K_part[0], FCM_part[0], DBSCAN_part[0], PA_part[0]])*(1/3)
	part2 = np.array([K_part[1], FCM_part[1], DBSCAN_part[1], PA_part[1]])*(1/3)
	part3 = np.array([K_part[2], FCM_part[2], DBSCAN_part[2], PA_part[2]])*(1/3)
	print("K_part: ", K_part, np.sum(K_part)*(1/3))
	print("FCM_part: ", FCM_part, np.sum(FCM_part)*(1/3))
	print("DBSCAN_part: ", DBSCAN_part, np.sum(DBSCAN_part)*(1/3))
	print("PA_part: ", PA_part, np.sum(PA_part)*(1/3), "\n")
	
	# Create the figure and axes
	fig, ax = plt.subplots(figsize=(8, 6))

	# Position of the bars on the x-axis
	bar_width = 0.5
	bar_positions = np.arange(len(categories))

	# Plotting each part of the bars
	ax.bar(bar_positions, part1, bar_width, label='1-cov', color="#5A5AAD")
	ax.bar(bar_positions, part2, bar_width, bottom=part1, label='2-cov', color="#E1C4C4")
	ax.bar(bar_positions, part3, bar_width, bottom=np.array(part1) + np.array(part2), label='3+-cov', color="#3C3C3C")

	# Create a second y-axis that shares the same x-axis
	ax2 = ax.twinx()

	# Copy the y-limits from the first axis
	ax2.set_ylim(ax.get_ylim())

	# Adding labels and title
	ax.set_ylabel('Performance', fontdict = {'fontsize': 20})
	ax.set_title('AMTkC for ' + condition + " targets", fontdict = {'fontsize': 20})
	ax.set_xticks(bar_positions)
	ax.set_xticklabels(categories, fontdict = {'fontsize': 18})
	ax.legend(loc = 'upper left', ncol = 1, shadow = True, fontsize = 13)

	# Setting the y-ticks and y-tick labels
	yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
	yticklabels = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticklabels, fontsize=20)  # Set the font size for y-tick labels
	ax2.set_yticks(yticks)
	ax2.set_yticklabels(yticklabels, fontsize=20)  # Set the font size for y-tick labels

	# Show the plot
	plt.show()

	my_path = "/home/leo/mts/src/QBSM/Extend/Data/KCOV"
	my_file = 'K-coverage_for_' + condition + "_targets"
	fig.savefig(os.path.join(my_path, my_file))

	return K_part, FCM_part, DBSCAN_part, PA_part

def Plot_Line(K_line, PA_line, FCM_line, DBSCAN_line):

	# print("K_line: ", K_line)
	# print("FCM_line: ", FCM_line)
	# print("DBSCAN_line: ", DBSCAN_line)
	# print("PA_line: ", PA_line)

	K_part = np.array([np.sum(element/3) for element in K_line])
	PA_part = np.array([np.sum(element/3) for element in PA_line])
	FCM_part = np.array([np.sum(element/3) for element in FCM_line])
	DBSCAN_part = np.array([np.sum(element/3) for element in DBSCAN_line])
	print("K_part: ", K_part)
	print("FCM_part: ", FCM_part)
	print("DBSCAN_part: ", DBSCAN_part)
	print("PA_part: ", PA_part)

	# Plot
	fig, ax = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-means', 1: 'K-means', 2: 'DBSk', 3: 'Hierarchical Method'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF", 3: "#613030"}
	Marker_Key = {0: "o", 1: "*", 2: "h", 3: "D"}

	x = np.array([4, 5, 6, 7, 8])
	x_ = np.array(["4", "5", "6", "7", "8"])
	y0 = K_part
	y1 = FCM_part
	y2 = DBSCAN_part
	y3 = PA_part

	ax.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax.plot(x, y3, color = Color_Key[3], label = Name_Key[3], marker = Marker_Key[3])

	# Create a second y-axis that shares the same x-axis
	ax2 = ax.twinx()

	# Copy the y-limits from the first axis
	ax2.set_ylim(ax.get_ylim())

	ax.set_title("AMTkC from 4-4 to 8-8 conditions", fontdict = {'fontsize': 20})
	ax.set_xlabel("Number of target v.s. observer", fontdict = {'fontsize': 20})
	ax.set_ylabel("Performance", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax.tick_params(axis='both', which='major', labelsize = 20)
	ax2.tick_params(axis='both', which='major', labelsize = 20)
	ax.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 14)

	plt.setp(ax, xticks=[4, 5, 6, 7, 8], xticklabels=["4", "5", "6", "7", "8"]
					, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], yticklabels=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"])
	plt.setp(ax2, xticks=[4, 5, 6, 7, 8], xticklabels=["4", "5", "6", "7", "8"]
					, yticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], yticklabels=["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"])
	plt.tight_layout()
	plt.show()

	# my_path = "/home/leo/mts/src/QBSM/Extend/Data/KCOV"
	# my_file = "KCOV_Line"
	# fig.savefig(os.path.join(my_path, my_file))

if __name__ == "__main__":

	filepath = "/home/leo/mts/src/QBSM/Extend/Data/KCOV"
	# Ob = ["Four/"]
	Ob = ["Four/", "Five/", "Six/", "Seven/", "Eight/"]
	# Times = ["10/"]
	# Times = ["1/", "2/", "3/", "4/", "5/"]
	Times = ["1/", "2/", "3/", "4/", "5/", "6/", "7/", "8/", "9/", "10/"]

	K_line, PA_line, FCM_line, DBSCAN_line = [], [], [], []

	for j, value in enumerate(Ob):

		K, PA, FCM, DBSCAN = [], [], [], []

		for i in Times:

			K_part = Plot_K(filepath, value, i); K.append(K_part)
			PA_part = Plot_PA(filepath, value, i); PA.append(PA_part)
			FCM_part = Plot_FCM(filepath, value, i); FCM.append(FCM_part)
			DBSCAN_part = Plot_DBSCAN(filepath, value, i); DBSCAN.append(DBSCAN_part)

		print("Condition: " + str(j+4))

		K_Data, FCM_Data, DBSCAN_Data, PA_Data = Plot_Bar(K, PA, FCM, DBSCAN, str(j+4))
		K_line.append(K_Data)
		PA_line.append(PA_Data)
		FCM_line.append(FCM_Data)
		DBSCAN_line.append(DBSCAN_Data)

	Plot_Line(K_line, PA_line, FCM_line, DBSCAN_line)