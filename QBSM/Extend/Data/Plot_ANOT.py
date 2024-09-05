import os
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure

def Plot_Sense(filepath, Type, Time):

	if Type[0] == "S":

		# FCM ---------------------------------------------------
		ANOT_FCM = []

		for i in range(5):
			
			Data_FCM, Time_FCM = [], []

			if i == 0:

				filename = filepath + "FCM/" + Time + Type + "10/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "FCM/" + Time + Type + "25/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "FCM/" + Time + Type + "50/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "FCM/" + Time + Type + "75/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "FCM/" + Time + Type + "90/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_FCM.append(data)
			Time_FCM.append(np.arange(1, 1001))

			Data_FCM = np.array(Data_FCM)
			Time_FCM = np.array(Time_FCM)

			g, g_k = 0, 0
			for j in range(np.shape(Data_FCM)[1]):

				for k in range(np.shape(Data_FCM)[2]):

					sum_ = 0

					for i in range(np.shape(Data_FCM)[0]):

						if Data_FCM[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_FCM)[1]
			M = np.shape(Data_FCM)[2]
			N = np.shape(Data_FCM)[0]
			ANOT = g/T/M
			ANOT_FCM.append(ANOT)
		print("ANOT_FCM: ", ANOT_FCM)
		# -----------------------------------------------

		# K ---------------------------------------------------
		ANOT_K = []

		for i in range(5):
			
			Data_K, Time_K = [], []

			if i == 0:

				filename = filepath + "K/" + Time + Type + "10/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "K/" + Time + Type + "25/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "K/" + Time + Type + "50/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "K/" + Time + Type + "75/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "K/" + Time + Type + "90/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_K.append(data)
			Time_K.append(np.arange(1, 1001))

			Data_K = np.array(Data_K)
			Time_K = np.array(Time_K)

			g, g_k = 0, 0
			for j in range(np.shape(Data_K)[1]):

				for k in range(np.shape(Data_K)[2]):

					sum_ = 0

					for i in range(np.shape(Data_K)[0]):

						if Data_K[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_K)[1]
			M = np.shape(Data_K)[2]
			N = np.shape(Data_K)[0]
			ANOT = g/T/M
			ANOT_K.append(ANOT)
		print("ANOT_K: ", ANOT_K)
		# -----------------------------------------------

		# DBSCAN ---------------------------------------------------
		ANOT_DBSCAN = []

		for i in range(5):
			
			Data_DBSCAN, Time_DBSCAN = [], []

			if i == 0:

				filename = filepath + "DBSCAN/" + Time + Type + "10/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "DBSCAN/" + Time + Type + "25/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "DBSCAN/" + Time + Type + "50/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "DBSCAN/" + Time + Type + "75/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "DBSCAN/" + Time + Type + "90/"
				files = Path(filename).glob("*.csv")

			exclude_file = "Eps.csv"

			for file in files:

				if file.name != exclude_file:

					data = []

					with open(file, "r", encoding='UTF8', newline='') as f:

						reader = csv.reader(f)
						temp = []
						
						for mem in reader:
							
							temp.append([float(element) for element in mem[0:len(mem)-1]])

						data = temp
						
					Data_DBSCAN.append(data)
			Time_DBSCAN.append(np.arange(1, 1001))

			Data_DBSCAN = np.array(Data_DBSCAN)
			Time_DBSCAN = np.array(Time_DBSCAN)

			g, g_k = 0, 0
			for j in range(np.shape(Data_DBSCAN)[1]):

				for k in range(np.shape(Data_DBSCAN)[2]):

					sum_ = 0

					for i in range(np.shape(Data_DBSCAN)[0]):

						if Data_DBSCAN[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_DBSCAN)[1]
			M = np.shape(Data_DBSCAN)[2]
			N = np.shape(Data_DBSCAN)[0]
			ANOT = g/T/M
			ANOT_DBSCAN.append(ANOT)
		print("ANOT_DBSCAN: ", ANOT_DBSCAN)
		# -----------------------------------------------

		# PA ---------------------------------------------------
		ANOT_PA = []

		for i in range(5):
			
			Data_PA, Time_PA = [], []

			if i == 0:

				filename = filepath + "PA/" + Time + Type + "10/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "PA/" + Time + Type + "25/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "PA/" + Time + Type + "50/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "PA/" + Time + Type + "75/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "PA/" + Time + Type + "90/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_PA.append(data)
			Time_PA.append(np.arange(1, 1001))

			Data_PA = np.array(Data_PA)
			Time_PA = np.array(Time_PA)

			g, g_k = 0, 0
			for j in range(np.shape(Data_PA)[1]):

				for k in range(np.shape(Data_PA)[2]):

					sum_ = 0

					for i in range(np.shape(Data_PA)[0]):

						if Data_PA[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_PA)[1]
			M = np.shape(Data_PA)[2]
			N = np.shape(Data_PA)[0]
			ANOT = g/T/M
			ANOT_PA.append(ANOT)
		print("ANOT_PA: ", ANOT_PA, "\n")
		# -----------------------------------------------

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'DBSk', 3: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF", 3: "#613030"}
	Marker_Key = {0: "o", 1: "*", 2: "h", 3: "D"}

	# init()

	x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_DBSCAN
	y3 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.plot(x, y3, color = Color_Key[3], label = Name_Key[3], marker = Marker_Key[3])
	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Target speed (Sensing range set at " + Type[2] + Type[3] + ") (unit/s)", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 12)

	plt.setp(ax0, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["0.1", "0.25", "0.5", "0.75", "0.9"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	return [ANOT_FCM, ANOT_K, ANOT_DBSCAN, ANOT_PA]

def Plot_Sense_Line(K, PA, FCM, DBSCAN, Type):

	Type = "0030"

	ANOT_FCM = np.mean(FCM, axis=0)
	ANOT_K = np.mean(K, axis=0)
	ANOT_DBSCAN = np.mean(DBSCAN, axis=0)
	ANOT_PA = np.mean(PA, axis=0) 
	print("Avg-----")
	print("ANOT_FCM: ", ANOT_FCM)
	print("ANOT_K: ", ANOT_K)
	print("ANOT_DBSCAN: ", ANOT_DBSCAN)
	print("ANOT_PA: ", ANOT_PA, "\n")

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-means', 1: 'K-means', 2: 'DBSk', 3: 'Hierarchical Method'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF", 3: "#613030"}
	Marker_Key = {0: "o", 1: "*", 2: "h", 3: "D"}

	# init()

	x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_DBSCAN
	y3 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.plot(x, y3, color = Color_Key[3], label = Name_Key[3], marker = Marker_Key[3])

	# Create a second y-axis that shares the same x-axis
	ax2 = ax0.twinx()

	# Copy the y-limits from the first axis
	ax2.set_ylim(ax0.get_ylim())

	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	# ax0.set_xlabel("Target speed (Sensing range set at " + Type[2] + Type[3] + ") (unit/s)", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Target speed (unit/s)", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax2.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 14)

	plt.setp(ax0, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["1", "2.5", "5", "7.5", "9"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.setp(ax2, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["1", "2.5", "5", "7.5", "9"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()

	plt.show()

	# my_path = "/home/leo/mts/src/QBSM/Extend/Data/ANOT"
	# my_file = "ANOT_S_30"
	# my_file = "ANOT_S_90"
	# fig0.savefig(os.path.join(my_path, my_file))

def Plot_Speed(filepath, Type, Time):

	if Type[0] == "T":

		# FCM ---------------------------------------------------
		ANOT_FCM = []

		for i in range(5):
			
			Data_FCM, Time_FCM = [], []

			if i == 0:

				filename = filepath + "FCM/" + Time + Type + "2/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "FCM/" + Time + Type + "3/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "FCM/" + Time + Type + "4/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "FCM/" + Time + Type + "5/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "FCM/" + Time + Type + "6/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_FCM.append(data)
			Time_FCM.append(np.arange(1, 1001))

			Data_FCM = np.array(Data_FCM)
			Time_FCM = np.array(Time_FCM)

			g, g_k = 0, 0
			for j in range(np.shape(Data_FCM)[1]):

				for k in range(np.shape(Data_FCM)[2]):

					sum_ = 0

					for i in range(np.shape(Data_FCM)[0]):

						if Data_FCM[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_FCM)[1]
			M = np.shape(Data_FCM)[2]
			N = np.shape(Data_FCM)[0]
			ANOT = g/T/M
			ANOT_FCM.append(ANOT)
		print("ANOT_FCM: ", ANOT_FCM)
		# -----------------------------------------------

		# K ---------------------------------------------------
		ANOT_K = []

		for i in range(5):
			
			Data_K, Time_K = [], []

			if i == 0:

				filename = filepath + "K/" + Time + Type + "2/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "K/" + Time + Type + "3/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "K/" + Time + Type + "4/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "K/" + Time + Type + "5/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "K/" + Time + Type + "6/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_K.append(data)
			Time_K.append(np.arange(1, 1001))

			Data_K = np.array(Data_K)
			Time_K = np.array(Time_K)

			g, g_k = 0, 0
			for j in range(np.shape(Data_K)[1]):

				for k in range(np.shape(Data_K)[2]):

					sum_ = 0

					for i in range(np.shape(Data_K)[0]):

						if Data_K[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_K)[1]
			M = np.shape(Data_K)[2]
			N = np.shape(Data_K)[0]
			ANOT = g/T/M
			ANOT_K.append(ANOT)
		print("ANOT_K: ", ANOT_K)
		# -----------------------------------------------

		# DBSCAN ---------------------------------------------------
		ANOT_DBSCAN = []

		for i in range(5):
			
			Data_DBSCAN, Time_DBSCAN = [], []

			if i == 0:

				filename = filepath + "DBSCAN/" + Time + Type + "2/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "DBSCAN/" + Time + Type + "3/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "DBSCAN/" + Time + Type + "4/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "DBSCAN/" + Time + Type + "5/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "DBSCAN/" + Time + Type + "6/"
				files = Path(filename).glob("*.csv")

			exclude_file = "Eps.csv"

			for file in files:

				if file.name != exclude_file:

					data = []

					with open(file, "r", encoding='UTF8', newline='') as f:

						reader = csv.reader(f)
						temp = []
						
						for mem in reader:
							
							temp.append([float(element) for element in mem[0:len(mem)-1]])

						data = temp
						
					Data_DBSCAN.append(data)
			Time_DBSCAN.append(np.arange(1, 1001))

			Data_DBSCAN = np.array(Data_DBSCAN)
			Time_DBSCAN = np.array(Time_DBSCAN)

			g, g_k = 0, 0
			for j in range(np.shape(Data_DBSCAN)[1]):

				for k in range(np.shape(Data_DBSCAN)[2]):

					sum_ = 0

					for i in range(np.shape(Data_DBSCAN)[0]):

						if Data_DBSCAN[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_DBSCAN)[1]
			M = np.shape(Data_DBSCAN)[2]
			N = np.shape(Data_DBSCAN)[0]
			ANOT = g/T/M
			ANOT_DBSCAN.append(ANOT)
		print("ANOT_DBSCAN: ", ANOT_DBSCAN)
		# -----------------------------------------------

		# PA ---------------------------------------------------
		ANOT_PA = []

		for i in range(5):
			
			Data_PA, Time_PA = [], []

			if i == 0:

				filename = filepath + "PA/" + Time + Type + "2/"
				files = Path(filename).glob("*.csv")
			elif i == 1:

				filename = filepath + "PA/" + Time + Type + "3/"
				files = Path(filename).glob("*.csv")
			elif i == 2:

				filename = filepath + "PA/" + Time + Type + "4/"
				files = Path(filename).glob("*.csv")
			elif i == 3:

				filename = filepath + "PA/" + Time + Type + "5/"
				files = Path(filename).glob("*.csv")
			elif i == 4:

				filename = filepath + "PA/" + Time + Type + "6/"
				files = Path(filename).glob("*.csv")

			for file in files:

				data = []

				with open(file, "r", encoding='UTF8', newline='') as f:

					reader = csv.reader(f)
					temp = []
					
					for mem in reader:
						
						temp.append([float(element) for element in mem[0:len(mem)-1]])

					data = temp
					
				Data_PA.append(data)
			Time_PA.append(np.arange(1, 1001))

			Data_PA = np.array(Data_PA)
			Time_PA = np.array(Time_PA)

			g, g_k = 0, 0
			for j in range(np.shape(Data_PA)[1]):

				for k in range(np.shape(Data_PA)[2]):

					sum_ = 0

					for i in range(np.shape(Data_PA)[0]):

						if Data_PA[i][j][k] == 1:

							sum_ = 1

					g += sum_


			T = np.shape(Data_PA)[1]
			M = np.shape(Data_PA)[2]
			N = np.shape(Data_PA)[0]
			ANOT = g/T/M
			ANOT_PA.append(ANOT)
		print("ANOT_PA: ", ANOT_PA, "\n")
		# -----------------------------------------------

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'DBSk', 3: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF", 3: "#613030"}
	Marker_Key = {0: "o", 1: "*", 2: "h", 3: "D"}

	# init()

	x = np.array([2, 3, 4, 5, 6])
	x_ = np.array(["2", "3", "4", "5", "6"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_DBSCAN
	y3 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.plot(x, y3, color = Color_Key[3], label = Name_Key[3], marker = Marker_Key[3])
	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Sensing range (Target speed set at " + Type[2] + ")", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 12)

	plt.setp(ax0, xticks=[2, 3, 4, 5, 6], xticklabels=["2", "3", "4", "5", "6"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	return [ANOT_FCM, ANOT_K, ANOT_DBSCAN, ANOT_PA]

def Plot_Speed_Line(K, PA, FCM, DBSCAN, Type):

	ANOT_FCM = np.mean(FCM, axis=0)
	ANOT_K = np.mean(K, axis=0)
	ANOT_DBSCAN = np.mean(DBSCAN, axis=0)
	ANOT_PA = np.mean(PA, axis=0) 
	print("Avg-----")
	print("ANOT_FCM: ", ANOT_FCM)
	print("ANOT_K: ", ANOT_K)
	print("ANOT_DBSCAN: ", ANOT_DBSCAN)
	print("ANOT_PA: ", ANOT_PA, "\n")

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-means', 1: 'K-means', 2: 'DBSk', 3: 'Hierarchical Method'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF", 3: "#613030"}
	Marker_Key = {0: "o", 1: "*", 2: "h", 3: "D"}

	# init()

	x = np.array([2, 3, 4, 5, 6])
	x_ = np.array(["2", "3", "4", "5", "6"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_DBSCAN
	y3 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.plot(x, y3, color = Color_Key[3], label = Name_Key[3], marker = Marker_Key[3])

	# Create a second y-axis that shares the same x-axis
	ax2 = ax0.twinx()

	# Copy the y-limits from the first axis
	ax2.set_ylim(ax0.get_ylim())

	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	# ax0.set_xlabel("Sensing range (Target speed set at " + Type[2] + ")", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Sensing range", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax2.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 14)

	plt.setp(ax0, xticks=[2, 3, 4, 5, 6], xticklabels=["30", "45", "60", "75", "90"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.setp(ax2, xticks=[2, 3, 4, 5, 6], xticklabels=["30", "45", "60", "75", "90"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	my_path = "/home/leo/mts/src/QBSM/Extend/Data/ANOT"
	# my_file = "ANOT_T_10"
	my_file = "ANOT_T_90"
	fig0.savefig(os.path.join(my_path, my_file))

def Plot_DBSCAN(filepath, Type, Time):

	# DBSCAN ---------------------------------------------------
	ANOT_DBSCAN = []

	for i in range(5):
		
		Data_DBSCAN, Time_DBSCAN = [], []

		if i == 0:

			filename = filepath + "DBSCAN/" + Time + Type + "10/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = filepath + "DBSCAN/" + Time + Type + "25/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = filepath + "DBSCAN/" + Time + Type + "50/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = filepath + "DBSCAN/" + Time + Type + "75/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = filepath + "DBSCAN/" + Time + Type + "90/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data = []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])

				data = temp
				
			Data_DBSCAN.append(data)
		Time_DBSCAN.append(np.arange(1, 1001))

		Data_DBSCAN = np.array(Data_DBSCAN)
		Time_DBSCAN = np.array(Time_DBSCAN)

		g, g_k = 0, 0
		for j in range(np.shape(Data_DBSCAN)[1]):

			for k in range(np.shape(Data_DBSCAN)[2]):

				sum_ = 0

				for i in range(np.shape(Data_DBSCAN)[0]):

					if Data_DBSCAN[i][j][k] == 1:

						sum_ = 1

				g += sum_


		T = np.shape(Data_DBSCAN)[1]
		M = np.shape(Data_DBSCAN)[2]
		N = np.shape(Data_DBSCAN)[0]
		ANOT = g/T/M
		ANOT_DBSCAN.append(ANOT)
	print("ANOT_DBSCAN: ", ANOT_DBSCAN)
	# -----------------------------------------------

	return ANOT_DBSCAN

if __name__ == "__main__":

	filepath = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/"

	Types = ["S_60/"]
	# Types = ["S_20/", "S_60/","T_10/", "T_90/"]

	# Times = ["Ten/"]
	# Times = ["One/", "Two/", "Three/", "Four/", "Five/"]
	Times = ["One/", "Two/", "Three/", "Four/", "Five/", "Six/", "Seven/", "Eight/", "Nine/", "Ten/"]

	for j, value in enumerate(Types):

		K, PA, FCM, DBSCAN = [], [], [], []

		if value[0] == "S":

			print(value[0] + value[1] + value[2] + "--------------------", "\n")

			for i, value_ in enumerate(Times):

				print("Times: ", i)
				ret = Plot_Sense(filepath, value, value_)
				FCM.append(ret[0])
				K.append(ret[1])
				DBSCAN.append(ret[2])
				PA.append(ret[3])

			# for i, value_ in enumerate(Times_10):

			# 	ret = Plot_DBSCAN(filepath, value, value_)
			# 	DBSCAN.append(ret)

			Plot_Sense_Line(K, PA, FCM, DBSCAN, value)
		elif value[0] == "T":

			print(value[0] + value[1] + value[2] + "--------------------", "\n")

			for i, value_ in enumerate(Times):

				print("Times: ", i)
				ret = Plot_Speed(filepath, value, value_)
				FCM.append(ret[0])
				K.append(ret[1])
				DBSCAN.append(ret[2])
				PA.append(ret[3])

			Plot_Speed_Line(K, PA, FCM, DBSCAN, value)
