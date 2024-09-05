import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':

	# p = {"0": 3, "1": 1, "2": 2}

	# for i in p:

	# 	print(p[i])

	# Define the arrays
	# array1 = np.array([1, 2, 3, 4, 5])
	# array2 = np.array([5, 4, 3, 2, 1])
	# array3 = np.array([2, 2, 2, 2, 2])

	# # Combine the arrays into a 2D array
	# combined_array = np.array([array1, array2, array3])

	# # Select the minimum element-wise among the arrays
	# result = np.min(combined_array, axis=0)

	# print(f"Result: {result}")

	# a = np.array([[1,2],1,3])
	# smallest_number = np.min(a)
	# indices = np.where(a == smallest_number)[0]
	# print(f"The indices of the smallest number are: {indices}")
	# print(len(indices))
	# a = np.array([1,1,3])
	# smallest_number = np.min(a)
	# print("smallest_number: ", smallest_number)

	# a = [2,1,5]
	# b = [2,3,1]
	# print(sorted(a) == sorted(b))
	# print(all(element == 0 for element in a))
	# print(np.where(a == 1))

	# a = np.array([2.1, 2.2])
	# b = np.array([0,0])
	# print("b: ", b)
	# print(np.zeros(2))
	# print(np.array_equal(a, b))
	# print(any(np.array_equal(a, row) for row in b))
	# print(np.vstack((b,a)))

	# v1 = np.array([14.4946, 12.4816])
	# v2 = np.array([14.8055, 12.7083])
	# v3 = np.array([15.0998, 12.4828])

	# I = np.array([[14.6043, 11.5434], [14.6344, 13.6354], [15.8611, 12.6345]])

	# d1 = np.linalg.norm(np.subtract(I, v1), axis=1)
	# print("d1: ", d1)
	# print("s1: ", np.argmin(d1))

	# d2 = np.linalg.norm(np.subtract(I, v2), axis=1)
	# print("d2: ", d2)
	# print("s2: ", np.argmin(d2))

	# d3 = np.linalg.norm(np.subtract(I, v3), axis=1)
	# print("d3: ", d3)
	# print("s3: ", np.argmin(d3))

	# c = {"1": 2, "3": 4, "5": 6}

	# for key, value in enumerate(c):

	# 	print(key)
	# 	print(value)

	# a = np.array([[5, 1, 5],
	# 			[5, 5, 1],
	# 			[1, 5, 5]])
	# a = np.array([[5, 1],
	# 			[1, 5],
	# 			[1, 5]])
	# row, col = linear_sum_assignment(a)
	# print("row: ", row)
	# print("col: ", col)

	# u_des = np.array([0.0,0.0,0.0])
	# u_mag = np.linalg.norm(u_des)
	# print("u_mag: ", u_mag)
	# np.min((1.0, u_mag))

	# vector = [1, 0]
	# angle = np.arctan2(vector[1], vector[0])

	# if angle < 0:

	# 	angle += 2 * np.pi
	# print(angle)

	# a = ["/uav1", "/uav3"]; a = set(a)
	# b = ["/uav1", "/uav3", "/uav4", "/uav5", "uav6"]; b = set(b)

	# print(a-b)
	# print(sorted(b-a))

	# a = ["/targetResdiual", "/CurrVel", "/SweetSpot", "/WinBid", "/TaskComplete", "/ResetReady",\
	# 		"/ResetComplete", "/Counting", "/Essential", "/Detection"]
	# for i, ref_topic in enumerate(a):
		
	# 	if ref_topic == "/Counting":
			
	# 		print(type(ref_topic))

	# a = [
	# {'4': np.array([[None, None]]), '1': np.array([[None, None]], dtype=object), '3': np.array([[None, None]], dtype=object), '2': np.array([[None, None]], dtype=object)},
	# {'4': np.array([0.10821742, 0.69689966]), '1': np.array([ 0.2233936 , -0.89778788]), '3': np.array([-0.29011171, -0.83938492]), '2': np.array([-0.28271814, -1.3191952 ])},
	# {'4': np.array([13.07225446,  9.30638453]), '1': np.array([11.75056831,  8.89630511]), '3': np.array([5.80534527, 8.58376252]), '2': np.array([5.77858785, 7.71197508])},
	# ]

	# b = copy.deepcopy(a)
	# print(b[0])

	# cost_matrix = np.array([[5, 2],
	# 			[5, 1],
	# 			[1, 5]])
	# row, col = linear_sum_assignment(cost_matrix)
	# print("row: ", row)
	# print("col: ", col)

	# n = len(cost_matrix)

	# sequence_num = list(range(0, n))
	# missing_numbers = [num for num in sequence_num if num not in row]
	# missing_numbers_hold = missing_numbers
	# print("missing_number: ", missing_numbers)

	# col_sol = {str(i): [] for i in range(n)}

	# for (row, col) in zip(row, col):

	# 	col_sol[str(row)] = col
	# print("col_sol: ", col_sol)

	# while len(missing_numbers) != 0:

	# 	cost_matrix_missing = np.array(cost_matrix[missing_numbers])
	# 	# print("cost_matrix_missing: ", cost_matrix_missing)

	# 	row_ind_missing, col_ind_missing = linear_sum_assignment(cost_matrix_missing)
	# 	# print("row_ind_missing: ", row_ind_missing)
	# 	# print("col_ind_missing: ", col_ind_missing)

	# 	for (row, col) in zip(missing_numbers, col_ind_missing):

	# 		col_sol[str(row)] = col
	# 		missing_numbers_hold.remove(row)

		
	# 	missing_numbers = missing_numbers_hold
	# print("col_sol: ", col_sol)
	# col_ind = np.zeros(n)
	# for key_, value_ in col_sol.items():

	# 	col_ind[int(key_)] = value_
	# print("col_ind: ", col_ind)

	a = np.array([[ 9.08457908, 13.49394843]])
	b = 1
	print(a is not None)