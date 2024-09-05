import os
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure
import matplotlib.animation as animation

plt.rcParams.update({'figure.max_open_warning': 0})

# Format plot
# fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))
# fig1, ax1 = plt.subplots(1, 1, figsize = (13.5, 10.125))
# fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(8, 1, figsize = (13.5, 10.125))
# fig, ax = plt.subplots(4, 2, figsize = (13.5, 10.125))


def init():

	ax0.clear()
	# ax1.clear()
	# ax2.clear()
	# ax3.clear()

	# ax[0,0].clear()
	# ax[1,0].clear()
	# ax[2,0].clear()
	# ax[3,0].clear()
	# ax[0,1].clear()
	# ax[1,1].clear()
	# ax[2,1].clear()
	# ax[3,1].clear()

def Read_Data_Target_Speed(Times_ = "One/", Type_ = "S_2"):

	Times = Times_
	Type = Type_

	K_coverage = "k"

	# FCM
	ANOT_FCM, K_FCM, PRE_FCM = [], [], []

	for i in range(5):

		Data_FCM, Time_FCM = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/10/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/25/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/50/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/75/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/90/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp

			# print("Time Step: " + str(np.shape(time_)))
		
			Data_FCM.append(data)
		Time_FCM.append(time_)

		Data_FCM = np.array(Data_FCM)
		Time_FCM = np.array(Time_FCM)

		g, g_k = 0, 0
		for j in range(np.shape(Data_FCM)[1]):

			for k in range(np.shape(Data_FCM)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_FCM)[0]):

					if Data_FCM[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		# print("g: " + str(g))
		T = np.shape(Data_FCM)[1]
		M = np.shape(Data_FCM)[2]
		N = np.shape(Data_FCM)[0]
		ANOT = g/T/M
		ANOT_FCM.append(ANOT)

		PRE_FCM.append(g)
		# K_FCM.append(g_k)
		perfomance = g_k/T/(M*N)
		K_FCM.append(perfomance)

		# print("ANOT_FCM: " + str(ANOT_FCM))
		# print("\n")

	# -----------------------------------------------

	# K
	ANOT_K, K_K, PRE_K = [], [], []
	
	for i in range(5):

		Data_K, Time_K = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/10/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/25/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/50/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/75/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/90/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp

			# print("Time Step: " + str(np.shape(time_)))
				
			Data_K.append(data)
		Time_K.append(time_)

		Data_K = np.array(Data_K)
		Time_K = np.array(Time_K)

		g, g_k = 0, 0
		for j in range(np.shape(Data_K)[1]):

			for k in range(np.shape(Data_K)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_K)[0]):

					if Data_K[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		# print("g: " + str(g))
		T = np.shape(Data_K)[1]
		M = np.shape(Data_K)[2]
		N = np.shape(Data_K)[0]
		ANOT = g/T/M
		ANOT_K.append(ANOT)

		PRE_K.append(g)
		# K_K.append(g_k)
		perfomance = g_k/T/(M*N)
		K_K.append(perfomance)
		# print("ANOT_K: " + str(ANOT_K))
		# print("\n")

	# -----------------------------------------------

	# PA
	ANOT_PA, K_PA, PRE_PA = [], [], []

	for i in range(5):

		# PA
		Data_PA, Time_PA = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/10/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/25/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/50/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/75/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/90/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp

			# print("Time Step: " + str(np.shape(time_)))
				
			Data_PA.append(data)
		Time_PA.append(time_)

		Data_PA = np.array(Data_PA)
		Time_PA = np.array(Time_PA)

		g, g_k = 0, 0
		for j in range(np.shape(Data_PA)[1]):

			for k in range(np.shape(Data_PA)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_PA)[0]):

					if Data_PA[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		# print("g: " + str(g))
		T = np.shape(Data_PA)[1]
		M = np.shape(Data_PA)[2]
		N = np.shape(Data_PA)[0]
		ANOT = g/T/M
		ANOT_PA.append(ANOT)

		PRE_PA.append(g)
		# K_PA.append(g_k)
		perfomance = g_k/T/(M*N)
		K_PA.append(perfomance)
		# print("ANOT_PA: " + str(ANOT_PA))
		# print("\n")

	# print("ANOT_FCM: " + str(ANOT_FCM))
	# print("ANOT_K: " + str(ANOT_K))
	# print("ANOT_PA: " + str(ANOT_PA), "\n")

	# print("PRE_FCM: " + str(PRE_FCM))
	# print("PRE_K: " + str(PRE_K))
	# print("PRE_PA: " + str(PRE_PA), "\n")

	# print("K_FCM: " + str(K_FCM))
	# print("K_K: " + str(K_K))
	# print("K_PA: " + str(K_PA), "\n")

	# for i in range(len(K_FCM)):

	# 	K_FCM[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])
	# 	K_K[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])
	# 	K_PA[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])

	# print("K_FCM: " + str(K_FCM))
	# print("K_K: " + str(K_K))
	# print("K_PA: " + str(K_PA))

	# -------------------------------------------------------------

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF"}
	Marker_Key = {0: "o", 1: "*", 2: "h"}

	# init()

	x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Target speed (Sensing range set at " + Type + ") (unit/s)", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 12)

	plt.setp(ax0, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["0.1", "0.25", "0.5", "0.75", "0.9"]
					, yticks=[0.25, 0.5, 0.75, 1.0], yticklabels=["0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	# my_path = "/home/leo/圖片/Research_Picture/9.13/"
	# my_file = 'ANOT_TimeSteps_Target_Speed.png'
	# fig0.savefig(os.path.join(my_path, my_file))

	# fig1, ax1 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	# x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	# x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	# y0 = K_FCM
	# y1 = K_K
	# y2 = K_PA

	# ax1.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	# ax1.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	# ax1.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	# ax1.set_title("Comparison of " + str(K_coverage) + "-coverage" + " of multiple targets", fontdict = {'fontsize': 20})
	# ax1.set_xlabel("Target speed (Sensing range set at 4) (unit/s)", fontdict = {'fontsize': 20})
	# ax1.set_ylabel(str(K_coverage) + "-coverage", fontdict = {'fontsize': 20})
	# # ax0.set_xticks(x, x_)
	# ax1.tick_params(axis='both', which='major', labelsize = 20)
	# ax1.legend(loc = 'upper right', ncol = 1, shadow = True, fontsize = 12)

	# plt.setp(ax1, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["0.1", "0.25", "0.5", "0.75", "0.9"]
	# 				, yticks=[0.10, 0.15, 0.20, 0.25], yticklabels=["0.10", "0.15", "0.20", "0.25"])
	# plt.tight_layout()
	# plt.show()

	# my_path = "/home/leo/圖片/Research_Picture/8.1/"
	# my_file = 'K_Coverage_Performancce_TimeSteps_Target_Speed.png'
	# fig1.savefig(os.path.join(my_path, my_file))

	return ANOT_FCM, ANOT_K, ANOT_PA

def Read_Data_Sensing_Range(Times_ = "One/", Type_ = "T_10"):

	Times = Times_
	Type = Type_

	K_coverage = "k"

	# FCM
	ANOT_FCM, K_FCM, PRE_FCM = [], [], []

	for i in range(5):
		
		Data_FCM, Time_FCM = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/2/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/3/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/4/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/5/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/FCM/Table/" + Times + Type + "/6/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp
				
			Data_FCM.append(data)
		Time_FCM.append(time_)

		Data_FCM = np.array(Data_FCM)
		Time_FCM = np.array(Time_FCM)

		g, g_k = 0, 0
		for j in range(np.shape(Data_FCM)[1]):

			for k in range(np.shape(Data_FCM)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_FCM)[0]):

					if Data_FCM[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		T = np.shape(Data_FCM)[1]
		M = np.shape(Data_FCM)[2]
		N = np.shape(Data_FCM)[0]
		ANOT = g/T/M
		ANOT_FCM.append(ANOT)

		# PRE_FCM.append(g)
		# K_FCM.append(g_k)
		perfomance = g_k/T/(M*N)
		K_FCM.append(perfomance)

	# -----------------------------------------------

	# K
	ANOT_K, K_K, PRE_K = [], [], []

	for i in range(5):

		Data_K, Time_K = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/2/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/3/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/4/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/5/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/K/Table/" + Times + Type + "/6/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp
				
			Data_K.append(data)
		Time_K.append(time_)

		Data_K = np.array(Data_K)
		Time_K = np.array(Time_K)

		g, g_k = 0, 0
		for j in range(np.shape(Data_K)[1]):

			for k in range(np.shape(Data_K)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_K)[0]):

					if Data_K[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		T = np.shape(Data_K)[1]
		M = np.shape(Data_K)[2]
		N = np.shape(Data_K)[0]
		ANOT = g/T/M
		ANOT_K.append(ANOT)

		# PRE_K.append(g)
		# K_K.append(g_k)
		perfomance = g_k/T/(M*N)
		K_K.append(perfomance)

	# -----------------------------------------------

	# PA
	ANOT_PA, K_PA, PRE_PA = [], [], []

	for i in range(5):

		Data_PA, Time_PA = [], []

		if i == 0:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/2/"
			files = Path(filename).glob("*.csv")
		elif i == 1:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/3/"
			files = Path(filename).glob("*.csv")
		elif i == 2:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/4/"
			files = Path(filename).glob("*.csv")
		elif i == 3:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/5/"
			files = Path(filename).glob("*.csv")
		elif i == 4:

			filename = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/PA/Table/" + Times + Type + "/6/"
			files = Path(filename).glob("*.csv")

		for file in files:

			data, time_ = [], []

			with open(file, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem[0:len(mem)-1]])
					time_.append(float(mem[-1]))

				data = temp
				
			Data_PA.append(data)
		Time_PA.append(time_)

		Data_PA = np.array(Data_PA)
		Time_PA = np.array(Time_PA)

		g, g_k = 0, 0
		for j in range(np.shape(Data_PA)[1]):

			for k in range(np.shape(Data_PA)[2]):

				sum_, sum_k = 0, 0

				for i in range(np.shape(Data_PA)[0]):

					if Data_PA[i][j][k] == 1:

						sum_ = 1
						sum_k += 1

				g += sum_
				g_k += sum_k

				# if sum_k >= K_coverage:

				# 	g_k += 1*sum_
				# else:

				# 	g_k += 0*sum_

		T = np.shape(Data_PA)[1]
		M = np.shape(Data_PA)[2]
		N = np.shape(Data_PA)[0]
		ANOT = g/T/M
		ANOT_PA.append(ANOT)

		# PRE_PA.append(g)
		# K_PA.append(g_k)
		perfomance = g_k/T/(M*N)
		K_PA.append(perfomance)

	# print("ANOT_K: " + str(ANOT_K))
	# print("ANOT_FCM: " + str(ANOT_FCM))
	# print("ANOT_PA: " + str(ANOT_PA))

	# print("PRE_FCM: " + str(PRE_FCM))
	# print("PRE_K: " + str(PRE_K))
	# print("PRE_PA: " + str(PRE_PA), "\n")

	# print("K_FCM: " + str(K_FCM))
	# print("K_K: " + str(K_K))
	# print("K_PA: " + str(K_PA), "\n")

	# for i in range(len(K_FCM)):

	# 	K_FCM[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])
	# 	K_K[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])
	# 	K_PA[i] /= np.max([PRE_FCM[i], PRE_K[i], PRE_PA[i]])

	# print("K_FCM: " + str(K_FCM))
	# print("K_K: " + str(K_K))
	# print("K_PA: " + str(K_PA))

	# ----------------------------------------------------------------------------

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF"}
	Marker_Key = {0: "o", 1: "*", 2: "h"}

	# init()

	x = np.array([2, 3, 4, 5, 6])
	x_ = np.array(["2", "3", "4", "5", "6"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.set_title("Comparison of normalized average number of observed target", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Sensing range (Target speed set at " + Type + ")", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 12)

	plt.setp(ax0, xticks=[2, 3, 4, 5, 6], xticklabels=["2", "3", "4", "5", "6"]
					, yticks=[0.25, 0.5, 0.75, 1.0], yticklabels=["0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()



	# my_path = "/home/leo/圖片/Research_Picture/9.13/"
	# my_file = 'ANOT_TimeSteps_Sensing_Range.png'
	# fig0.savefig(os.path.join(my_path, my_file))

	# fig1, ax1 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	# x = np.array([3, 4, 5, 6, 7])
	# x_ = np.array(["3", "4", "5", "6", "7"])
	# y0 = K_FCM
	# y1 = K_K
	# y2 = K_PA

	# ax1.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	# ax1.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	# ax1.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	# ax1.set_title("Comparison of " + str(K_coverage) + "-coverage" + " of multiple targets", fontdict = {'fontsize': 20})
	# ax1.set_xlabel("Sensing range (Target speed set at 0.5)", fontdict = {'fontsize': 20})
	# ax1.set_ylabel(str(K_coverage) + "-coverage", fontdict = {'fontsize': 20})
	# # ax0.set_xticks(x, x_)
	# ax1.tick_params(axis='both', which='major', labelsize = 20)
	# ax1.legend(loc = 'upper right', ncol = 1, shadow = True, fontsize = 12)

	# plt.setp(ax1, xticks=[3, 4, 5, 6, 7], xticklabels=["3", "4", "5", "6", "7"]
	# 				, yticks=[0.10, 0.20, 0.30, 0.40], yticklabels=["0.10", "0.20", "0.30", "0.40"])
	# plt.tight_layout()
	# plt.show()

	# my_path = "/home/leo/圖片/Research_Picture/8.1/"
	# my_file = 'K_Coverage_Performance_TimeSteps_Sensing_Range.png'
	# fig1.savefig(os.path.join(my_path, my_file))

	return ANOT_FCM, ANOT_K, ANOT_PA

def Execute_read():

	# ANOT_K, ANOT_FCM, ANOT_PA = Read_Data_Target_Speed()
	ANOT_K, ANOT_FCM, ANOT_PA = Read_Data_Sensing_Range()

	Name_Key = {0: 'K-Means', 1: 'Fuzzy C-Means', 2: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF"}

	init()
	# for i in range(np.shape(x)[0]):

	# 	globals()["x0_"] = x[i]
	# 	globals()["y0_"] = ANOT_K[i]
	# 	globals()["x1_"] = x[i]
	# 	globals()["y2_"] = ANOT_FCM[i]
	# 	globals()["x2_"] = x[i]
	# 	globals()["y2_"] = ANOT_PA[i]

	# 	globals()["ax" + str(i)].plot(globals()["x_" + str(i)], globals()["y_" + str(i)], color = Color_Key[i])
	# 	globals()["ax" + str(i)].set_title(Name_Key[i], fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].set_ylabel("Probability", fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].tick_params(axis='both', which='major', labelsize = 20)

	x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	y0 = ANOT_K
	y1 = ANOT_FCM
	y2 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2])
	ax0.set_title("Comparison of Normalized Average Number of Observed Target", fontdict = {'fontsize': 20})
	ax0.set_xlabel("Speed Targets", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 20})
	# ax0.set_xticks(x, x_)
	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower left', ncol = 1, shadow = True, fontsize = 12)

	plt.setp(ax0, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["0.1", "0.25", "0.5", "0.75", "0.9"]
					, yticks=[0.25, 0.5, 0.75, 1.0], yticklabels=["0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	# my_path = "/home/leo/圖片/Research_Picture/8.1/"
	# my_file = 'ANOT_Target_Speed.png'
	# fig.savefig(os.path.join(my_path, my_file))

	# Name_Key = {0: 'Observation Score of Target 0', 1: 'Observation Score of Target 1', 2: 'Observation Score of Target 2',
	# 			3: 'Observation Score of Target 3', 4: 'Observation Score of Target 4', 5: 'Observation Score of Target 5',
	# 			6: 'Observation Score of Target 6', 7: 'Observation Score of Target 7'}

	# for i in range(np.shape(S)[0]):

	# 	globals()["x_" + str(i)] = time[i]
	# 	globals()["y_" + str(i)] = S[i]

	# 	ax[i%4, int(i/4)].plot(globals()["x_" + str(i)], globals()["y_" + str(i)], color = Color_Key[i])
	# 	ax[i%4, int(i/4)].set_title(Name_Key[i], fontdict = {'fontsize': 20})
	# 	ax[i%4, int(i/4)].set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# 	ax[i%4, int(i/4)].set_ylabel("Score", fontdict = {'fontsize': 20})
	# 	ax[i%4, int(i/4)].tick_params(axis='both', which='major', labelsize = 20)

	# plt.tight_layout()
	# plt.show()

	# my_path = "/home/leo/圖片/Research_Picture/7.18/Target_8_wA_Score"
	# my_file = 'Score_8_woA.png'
	# fig.savefig(os.path.join(my_path, my_file))


	# ax0.plot(x_0, y_0, color = Color_Key[0])
	# ax0.set_title(Name_Key[0], fontdict = {'fontsize': 20})
	# ax0.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# ax0.set_ylabel("Probability", fontdict = {'fontsize': 20})
	# ax0.tick_params(axis='both', which='major', labelsize = 20)

	# ax1.plot(x_1, y_1, color = "#272727")
	# ax1.set_title('Joint Probability of Target 1', fontdict = {'fontsize': 20})
	# ax1.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# ax1.set_ylabel("Probability", fontdict = {'fontsize': 20})
	# ax1.tick_params(axis='both', which='major', labelsize = 20)

	# ax2.plot(x_2, y_2, color = "#3C3C3C")
	# ax2.set_title('Joint Probability of Target 2', fontdict = {'fontsize': 20})
	# ax2.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# ax2.set_ylabel("Probability", fontdict = {'fontsize': 20})
	# ax2.tick_params(axis='both', which='major', labelsize = 20)

	# ax3.plot(x_3, y_3, color = "#4F4F4F")
	# ax3.set_title('Joint Probability of Target 3', fontdict = {'fontsize': 20})
	# ax3.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# ax3.set_ylabel("Probability", fontdict = {'fontsize': 20})
	# ax3.tick_params(axis='both', which='major', labelsize = 20)

	# plt.subplots_adjust(left = 0.2, bottom = 0.2, right = 0.8, top = 0.9, wspace = 0, hspace = 1.0)

	# figure(figsize = (10, 7.5), dpi = 80)
	# parameters = {'legend.fontsize': 15, "xtick.labelsize": 25, "ytick.labelsize": 25,
	# 				"axes.labelsize": 30, "axes.titlesize": 20}
	# plt.rcParams.update(parameters)

	# plt.cla()
	# plt.grid()
	# plt.plot(x, y, linewidth = 1.5, linestyle = '-', label = 'Cost')
	
	# plt.title('Cost Function of Agent 0')
	# plt.xlabel('Time (s)')
	# plt.ylabel('Cost')

	# plt.legend()

def PlotData():

	Data_Comparison, time_Comparison = [], []

	filename = "/home/leo/mts/src/QBSM/Data/Joint/Comparison"
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint, time_ = [], []
		data = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])
				time_.append(float(mem[-1]))

			data = Joint
			
		Data_Comparison.append(data)
		time_Comparison.append(time_)

	Data_Comparison = np.array(Data_Comparison)
	time_Comparison = np.array(time_Comparison)

	for k in range(np.shape(Data_Comparison)[2]):

		globals()["S_" + str(k)] = []
		globals()["P_" + str(k)] = []

	for j in range(np.shape(Data_Comparison)[1]):

		for k in range(np.shape(Data_Comparison)[2]):

			p = 1.0
			score = 0.0

			for i in range(np.shape(Data_Comparison)[0]):

				score += Data_Comparison[i][j][k]
				p *= 1 - Data_Comparison[i][j][k]

			P = 1 - p
			globals()["P_" + str(k)].append(P)

			score /= 1
			globals()["S_" + str(k)].append(score)

	S = []
	P = []
	for k in range(np.shape(Data_Comparison)[2]):

		S.append(globals()["S_" + str(k)])
		P.append(globals()["P_" + str(k)])

	S_total_comparison = np.sum(S, axis = 0)
	P_total_comparison = np.sum(P, axis = 0)

	# ----------------------------------------- #

	Data_Test, time_Test = [], []

	filename = "/home/leo/mts/src/QBSM/Data/Joint/Test"
	files = Path(filename).glob("*.csv")

	for file in files:

		Joint, time_ = [], []
		data = []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
			
			for mem in reader:
				
				Joint.append([float(element) for element in mem[0:len(mem)-1]])
				time_.append(float(mem[-1]))

			data = Joint
			
		Data_Test.append(data)
		time_Test.append(time_)

	Data_Test = np.array(Data_Test)
	time_Test = np.array(time_Test)

	for k in range(np.shape(Data_Test)[2]):

		globals()["S_" + str(k)] = []
		globals()["P_" + str(k)] = []

	for j in range(np.shape(Data_Test)[1]):

		for k in range(np.shape(Data_Test)[2]):

			p = 1.0
			score = 0.0

			for i in range(np.shape(Data_Test)[0]):

				score += Data_Test[i][j][k]
				p *= 1 - Data_Comparison[i][j][k]

			P = 1 - p
			globals()["P_" + str(k)].append(P)

			score /= 1
			globals()["S_" + str(k)].append(score)

	S = []
	P = []
	for k in range(np.shape(Data_Test)[2]):

		S.append(globals()["S_" + str(k)])
		P.append(globals()["P_" + str(k)])

	S_total_test = np.sum(S, axis = 0)
	P_total_test = np.sum(P, axis = 0)

	fig, ax = plt.subplots(1, 1, figsize = (13.5, 10.125))
	ax.clear()

	x_0 = time_Comparison[0]
	y_0 = S_total_comparison
	x_1 = time_Test[0]
	y_1 = S_total_test

	ax.plot(x_0, y_0, color = "#0000E3", label="w/o Algorithm")
	ax.plot(x_1, y_1, color = "#FF0000", label="w/ Algorithm")
	ax.set_title("Total Score of Team", fontdict = {'fontsize': 20})
	# ax.set_title("Total Joint Probability of Targets", fontdict = {'fontsize': 20})
	ax.set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	ax.set_ylabel("Score", fontdict = {'fontsize': 20})
	# ax.set_ylabel("Probability", fontdict = {'fontsize': 20})
	ax.tick_params(axis='both', which='major', labelsize = 20)
	ax.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 20)

	plt.tight_layout()
	plt.show()

	my_path = "/home/leo/圖片/Research_Picture/7.18/Target_8_Score/"
	my_file = 'Total_Score_sweetspot.png'
	fig.savefig(os.path.join(my_path, my_file))

def animate(i):

	Data = readData()

	for (mem, i) in zip(Data, range(len(Data))):

		globals()["x_" + str(i)] = np.ndarray.tolist(np.arange(0, np.shape(mem)[1], 1))
		globals()["y_" + str(i) + str(1)] = mem[0]
		globals()["y_" + str(i) + str(2)] = mem[1]
		globals()["y_" + str(i) + str(3)] = mem[2]

	init()

	if y_01 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax0.plot([], [], color = "#005AB5", label = lables_1)
		ax0.plot([], [], color = "#FF9224", label = lables_2)
		ax0.plot([], [], color = "#007500", label = lables_3)
		ax0.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_01[-1] >= 0) and (y_02[-1] >= 0) and (y_03[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_01[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_02[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_03[-1], 3))
		ax0.plot(x_0, y_01, color = "#005AB5", label = lables_1)
		ax0.plot(x_0, y_02, color = "#FF9224", label = lables_2)
		ax0.plot(x_0, y_03, color = "#007500", label = lables_3)
		ax0.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax0.set_title('Cost Finction of Agent 0 (Red)')
	ax0.set_xlabel("Instant")
	ax0.set_ylabel("Cost")
	ax0.set_ylim((0, 20))

	if y_11 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax1.plot([], [], color = "#005AB5", label = lables_1)
		ax1.plot([], [], color = "#FF9224", label = lables_2)
		ax1.plot([], [], color = "#007500", label = lables_3)
		ax1.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_11[-1] >= 0) and (y_12[-1] >= 0) and (y_13[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_11[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_12[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_13[-1], 3))
		ax1.plot(x_1, y_11, color = "#005AB5", label = lables_1)
		ax1.plot(x_1, y_12, color = "#FF9224", label = lables_2)
		ax1.plot(x_1, y_13, color = "#007500", label = lables_3)
		ax1.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax1.set_title('Cost Finction of Agent 1 (Green)')
	ax1.set_xlabel("Instant")
	ax1.set_ylabel("Cost")
	ax1.set_ylim((0, 20))

	if y_21 == []:

		lables_1 = "Cost Function 1, " + "Value: " + str(0)
		lables_2 = "Cost Function 2, " + "Value: " + str(0)
		lables_3 = "Cost Function 3, " + "Value: " + str(0)
		ax2.plot([], [], color = "#005AB5", label = lables_1)
		ax2.plot([], [], color = "#FF9224", label = lables_2)
		ax2.plot([], [], color = "#007500", label = lables_3)
		ax2.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)
	elif (y_21[-1] >= 0) and (y_22[-1] >= 0) and (y_23[-1] >= 0):

		lables_1 = "Cost Function 1, " + "Value: " + str(np.round(y_21[-1], 3))
		lables_2 = "Cost Function 2, " + "Value: " + str(np.round(y_22[-1], 3))
		lables_3 = "Cost Function 3, " + "Value: " + str(np.round(y_23[-1], 3))
		ax2.plot(x_2, y_21, color = "#005AB5", label = lables_1)
		ax2.plot(x_2, y_22, color = "#FF9224", label = lables_2)
		ax2.plot(x_2, y_23, color = "#007500", label = lables_3)
		ax2.legend(loc = "upper right", bbox_to_anchor = (1.25, 0.7), ncol = 1)

	ax2.set_title('Cost Finction of Agent 2 (Blue)')
	ax2.set_xlabel("Instant")
	ax2.set_ylabel("Cost")
	ax2.set_ylim((0, 20))

	# plt.tight_layout()
	plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = 0, hspace = 0.45)
	parameters = {'legend.fontsize': 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
					"axes.labelsize": 15, "axes.titlesize": 15}
	plt.rcParams.update(parameters)

def PlotAverage():

	Times_key = {1: "One/", 2: "Two/", 3: "Three/", 4: "Four/", 5: "Five/"}
	Type_key = {10: "T_10", 90: "T_90", 2: "S_2", 6: "S_6"}

	# Challenging Situations - ANOT Target Speed 0.90
	# ANOT_FCM, ANOT_K, ANOT_PA = [], [], []
	# for i in range(1,6):

	# 	FCM, K, PA = Read_Data_Sensing_Range(Times_ = Times_key[i], Type_ = Type_key[90])
	# 	ANOT_FCM.append(FCM)
	# 	ANOT_K.append(K)
	# 	ANOT_PA.append(PA)

	# ANOT_FCM = np.mean(np.array(ANOT_FCM), axis = 0)
	# ANOT_K = np.mean(np.array(ANOT_K), axis = 0)
	# ANOT_PA = np.mean(np.array(ANOT_PA), axis = 0)

	# Suitable Situations - ANOT Target Speed 0.10
	ANOT_FCM, ANOT_K, ANOT_PA = [], [], []
	for i in range(1,6):

		FCM, K, PA = Read_Data_Sensing_Range(Times_ = Times_key[i], Type_ = Type_key[10])
		ANOT_FCM.append(FCM)
		ANOT_K.append(K)
		ANOT_PA.append(PA)

	ANOT_FCM = np.mean(np.array(ANOT_FCM), axis = 0)
	ANOT_K = np.mean(np.array(ANOT_K), axis = 0)
	ANOT_PA = np.mean(np.array(ANOT_PA), axis = 0)

	# Plot
	fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'Proposed Algorithm'}
	Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF"}
	Marker_Key = {0: "o", 1: "*", 2: "h"}

	x = np.array([2, 3, 4, 5, 6])
	x_ = np.array(["2", "3", "4", "5", "6"])
	y0 = ANOT_FCM
	y1 = ANOT_K
	y2 = ANOT_PA

	ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	ax0.set_title("Comparison of Normalized ANOT", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})
	ax0.set_xlabel("Sensing range (units)", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})
	# ax0.set_xlabel("Varing the sensing range for the target speed set at 9 unit per time step", fontdict = {'fontsize': 20})
	ax0.set_ylabel("ANOT", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})

	ax0.tick_params(axis='both', which='major', labelsize = 20)
	ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 15)

	plt.setp(ax0, xticks=[2, 3, 4, 5, 6], xticklabels=["20", "30", "40", "50", "60"]
					, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	plt.tight_layout()
	plt.show()

	my_path = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/"
	my_file = 'ANOT_Target_Speed_1.png'
	fig0.savefig(os.path.join(my_path, my_file))

	# --------------------------------------------------------------------------------------------------------------------------#

	# Challenging Situations - ANOT Sensing Range 2
	# ANOT_FCM, ANOT_K, ANOT_PA = [], [], []
	# for i in range(1,6):

	# 	FCM, K, PA = Read_Data_Target_Speed(Times_ = Times_key[i], Type_ = Type_key[2])
	# 	ANOT_FCM.append(FCM)
	# 	ANOT_K.append(K)
	# 	ANOT_PA.append(PA)

	# ANOT_FCM = np.mean(np.array(ANOT_FCM), axis = 0)
	# ANOT_K = np.mean(np.array(ANOT_K), axis = 0)
	# ANOT_PA = np.mean(np.array(ANOT_PA), axis = 0)

	# Suitable Situations - ANOT Sensing Range 6
	# ANOT_FCM, ANOT_K, ANOT_PA = [], [], []
	# for i in range(1,6):

	# 	FCM, K, PA = Read_Data_Target_Speed(Times_ = Times_key[i], Type_ = Type_key[6])
	# 	ANOT_FCM.append(FCM)
	# 	ANOT_K.append(K)
	# 	ANOT_PA.append(PA)

	# ANOT_FCM = np.mean(np.array(ANOT_FCM), axis = 0)
	# ANOT_K = np.mean(np.array(ANOT_K), axis = 0)
	# ANOT_PA = np.mean(np.array(ANOT_PA), axis = 0)

	# Plot
	# fig0, ax0 = plt.subplots(1, 1, figsize = (13.5, 10.125))

	# Name_Key = {0: 'Fuzzy C-Means', 1: 'K-Means', 2: 'Proposed Algorithm'}
	# Color_Key = {0: "#FF0000", 1: "#00BB00", 2: "#2828FF"}
	# Marker_Key = {0: "o", 1: "*", 2: "h"}

	# x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
	# x_ = np.array(["0.1", "0.25", "0.5", "0.75", "0.9"])
	# y0 = ANOT_FCM
	# y1 = ANOT_K
	# y2 = ANOT_PA

	# ax0.plot(x, y0, color = Color_Key[0], label = Name_Key[0], marker = Marker_Key[0])
	# ax0.plot(x, y1, color = Color_Key[1], label = Name_Key[1], marker = Marker_Key[1])
	# ax0.plot(x, y2, color = Color_Key[2], label = Name_Key[2], marker = Marker_Key[2])
	# ax0.set_title("Comparison of Normalized ANOT", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})
	# ax0.set_xlabel("Target speed (unit per time step) ", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})
	# # ax0.set_xlabel("Target speed (Sensing range set at 60 units) unit per time step", fontdict = {'fontsize': 20})
	# ax0.set_ylabel("ANOT", fontdict = {'fontsize': 25, "fontname": "Times New Roman"})
	# # ax0.set_xticks(x, x_)
	# ax0.tick_params(axis='both', which='major', labelsize = 20)
	# ax0.legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 15)

	# plt.setp(ax0, xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xticklabels=["1", "2.5", "5", "7.5", "9"]
	# 				, yticks=[0.0, 0.25, 0.5, 0.75, 1.0], yticklabels=["0.0", "0.25", "0.5", "0.75", "1.0"])
	# plt.tight_layout()
	# plt.show()

	# my_path = "C:/Users/leomd/IME/Master Thesis/Lab ver/CTO-main/Data/ANOT/"
	# my_file = 'ANOT_Sensing_Range_6.png'
	# fig0.savefig(os.path.join(my_path, my_file))

	# print("ANOT_FCM: ", ANOT_FCM)
	# print("ANOT_K: ", ANOT_K)
	# print("ANOT_PA: ", ANOT_PA)

if __name__ == "__main__":

	# ani = animation.FuncAnimation(plt.gcf(), animate, cache_frame_data = False,\
	# 								init_func = init, interval = 30, blit = False)
	# plt.show()

	# Execute_read()
	# PlotData()
	# Read_Data_Target_Speed(Times_ = "One/", Type_ = "S_2")
	# Read_Data_Sensing_Range(Times_ = "One/", Type_ = "T_10")
	PlotAverage()