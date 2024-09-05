import csv
import numpy as np


def Determine_A_C(Times, Types, detailed):

	for detail in detailed:

		Mean, Variance, STD = [], [], []

		for i, time in enumerate(Times):

			filepath = "/home/leo/mts/src/QBSM/Extend/Data/ANOT/DBSCAN/" + time + Types[0] + detail + "/Eps.csv"

			data = []

			with open(filepath, "r", encoding='UTF8', newline='') as f:

				reader = csv.reader(f)
				temp = []
				
				for mem in reader:
					
					temp.append([float(element) for element in mem])

				data = temp

			print(time[0:len(time)-1])
			print("Average, Variance, Standard Deviation: ", np.mean(data), np.var(data), np.std(data), "\n")
			Mean.append(np.mean(data))
			Variance.append(np.var(data))
			STD.append(np.std(data))

		print(Types[0] + "" + detail + "Total average, variance, standard deviation: ", np.mean(Mean), np.mean(Variance), np.mean(STD), "\n")


if __name__ == '__main__':

	Types = ["S_60/"]
	# Types = ["S_20/", "S_60/","T_10/", "T_90/"]

	# detailed = ["10/"]
	detailed = ["10/", "25/", "50/", "75/", "90/"]
	# detailed = ["2", "3", "4", "5", "6"]

	Times = ["One/", "Two/", "Three/", "Four/", "Five/", "Six/", "Seven/", "Eight/", "Nine/", "Ten/"]

	Determine_A_C(Times, Types, detailed)