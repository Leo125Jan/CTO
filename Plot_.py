import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.pyplot import figure

def read_one_Data():

	C1, C2, C3, time_ = [], [], [], []
	Data = []

	filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/Data_0.csv"

	with open(filename, "r", encoding='UTF8', newline='') as f:

		reader = csv.reader(f)
		
		for mem in reader:
			
			C1.append(float(mem[0]))
			C2.append(float(mem[1]))
			C3.append(float(mem[2]))
			time_.append(float(mem[3]))

		Data = [C1[84:], C2[84:], C3[84:], time_[84:]]

	return Data

def readData():

	Data = []
	count = 0

	filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
	files = Path(filename).glob("*.csv")

	for file in files:

		data_ = []
		C1, C2, C3, time_ = [], [], [], []

		with open(file, "r", encoding='UTF8', newline='') as f:

			reader = csv.reader(f)
		
			for mem in reader:
				
				C1.append(float(mem[0]))
				C2.append(float(mem[1]))
				C3.append(float(mem[2]))
				time_.append(float(mem[3]))

			if count == 0:

				data_ = [C1[84:], C2[84:], C3[84:], time_[84:]]
			elif count == 1:

				data_ = [C1[67:], C2[67:], C3[67:], time_[67:]]
			elif count == 2:

				data_ = [C1[27:], C2[27:], C3[27:], time_[27:]]

		count += 1

		Data.append(data_)

	return Data

if __name__ == "__main__":

	Data = readData()
	count = 0

	for data in Data:

		x = data[3]
		y1 = data[0]
		y2 = data[1]
		y3 = data[2]

		figure(figsize = (10, 7.5), dpi = 80)

		plt.cla()
		plt.grid()
		plt.tight_layout()
		plt.plot(x, y1, linewidth = 1.5, linestyle = '-', label = 'Cost Function 1')
		plt.plot(x, y2, linewidth = 1.5, linestyle = '-', label = 'Cost Function 2')
		plt.plot(x, y3, linewidth = 1.5, linestyle = '-', label = 'Cost Function 3')
		
		plt.title('Cost Function of Agent ' + str(count))
		plt.xlabel('Time (s)')
		plt.ylabel('Cost')
		
		plt.legend()
		plt.show()

		count += 1