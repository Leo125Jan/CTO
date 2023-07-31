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
# fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize = (13.5, 10.125))
# fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(8, 1, figsize = (13.5, 10.125))
fig, ax = plt.subplots(4, 2, figsize = (13.5, 10.125))


def init():

	# ax0.clear()
	# ax1.clear()
	# ax2.clear()
	# ax3.clear()

	ax[0,0].clear()
	ax[1,0].clear()
	ax[2,0].clear()
	ax[3,0].clear()
	ax[0,1].clear()
	ax[1,1].clear()
	ax[2,1].clear()
	ax[3,1].clear()

def Read_Data():

	Data_Comparison, time_Comparison = [], []

	filename = "/home/leo/mts/src/QBSM/Data/Joint/Comparison/"
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

		globals()["P_" + str(k)] = []
		# globals()["S_" + str(k)] = []

	for j in range(np.shape(Data_Comparison)[1]):

		# score = 0.0

		for k in range(np.shape(Data_Comparison)[2]):

			p = 1.0

			for i in range(np.shape(Data_Comparison)[0]):

				p *= 1 - Data_Comparison[i][j][k]
				# score += Data_Comparison[i][j][k]

			P = 1 - p
			# score /= 1
			globals()["P_" + str(k)].append(P)
			# globals()["S_" + str(k)].append(score)

	# P = [P_0, P_1, P_2, P_3]
	P_Comparsion = []
	# S = []
	for k in range(np.shape(Data_Comparison)[2]):

		P_Comparsion.append(globals()["P_" + str(k)])
		# S.append(globals()["S_" + str(k)])

	# S_total = np.sum(S, axis = 0)

	# -------------------------------- #
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

		globals()["P_" + str(k)] = []

	for j in range(np.shape(Data_Test)[1]):


		for k in range(np.shape(Data_Test)[2]):

			p = 1.0

			for i in range(np.shape(Data_Test)[0]):

				p *= 1 - Data_Comparison[i][j][k]

			P = 1 - p
			globals()["P_" + str(k)].append(P)

	P_Test = []
	for k in range(np.shape(Data_Test)[2]):

		P_Test.append(globals()["P_" + str(k)])

	return P_Test, time_Test, P_Comparsion, time_Comparison

def Execute_read():

	P_Test, time_Test, P_Comparsion, time_Comparison = Read_Data()
	Name_Key = {0: 'Joint Probability of Target 0', 1: 'Joint Probability of Target 1', 2: 'Joint Probability of Target 2',
				3: 'Joint Probability of Target 3', 4: 'Joint Probability of Target 4', 5: 'Joint Probability of Target 5',
				6: 'Joint Probability of Target 6', 7: 'Joint Probability of Target 7'}
	Color_Key = {0: "#000000", 1: "#272727", 2: "#3C3C3C", 3: "#4F4F4F", 4: "#5B5B5B", 5: "#6C6C6C", 6: "#7B7B7B", 7: "#8E8E8E"}

	init()
	# for i in range(np.shape(P)[0]):

	# 	globals()["x_" + str(i)] = time[i]
	# 	globals()["y_" + str(i)] = P[i]

	# 	globals()["ax" + str(i)].plot(globals()["x_" + str(i)], globals()["y_" + str(i)], color = Color_Key[i])
	# 	globals()["ax" + str(i)].set_title(Name_Key[i], fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].set_xlabel("Time (s)", fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].set_ylabel("Probability", fontdict = {'fontsize': 20})
	# 	globals()["ax" + str(i)].tick_params(axis='both', which='major', labelsize = 20)

	for i in range(np.shape(P_Test)[0]):

		globals()["x0_" + str(i)] = time_Comparison[i]
		globals()["y0_" + str(i)] = P_Comparsion[i]
		globals()["x1_" + str(i)] = time_Test[i]
		globals()["y1_" + str(i)] = P_Test[i]

		ax[i%4, int(i/4)].plot(globals()["x0_" + str(i)], globals()["y0_" + str(i)], color = "#0000E3", label="w/o Algorithm")
		ax[i%4, int(i/4)].plot(globals()["x1_" + str(i)], globals()["y1_" + str(i)], color = "#FF0000", label="w/ Algorithm")
		ax[i%4, int(i/4)].set_title(Name_Key[i], fontdict = {'fontsize': 20})
		ax[i%4, int(i/4)].set_xlabel("Time (s)", fontdict = {'fontsize': 20})
		ax[i%4, int(i/4)].set_ylabel("Probability", fontdict = {'fontsize': 20})
		ax[i%4, int(i/4)].tick_params(axis='both', which='major', labelsize = 20)
		ax[i%4, int(i/4)].legend(loc = 'lower right', ncol = 1, shadow = True, fontsize = 12)

	plt.tight_layout()
	plt.show()

	my_path = "/home/leo/圖片/Research_Picture/7.18/Target_8_Joint_Probability"
	my_file = 'Joint_Probability_8.png'
	fig.savefig(os.path.join(my_path, my_file))

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

if __name__ == "__main__":

	# ani = animation.FuncAnimation(plt.gcf(), animate, cache_frame_data = False,\
	# 								init_func = init, interval = 30, blit = False)
	# plt.show()

	# Execute_read()
	PlotData()