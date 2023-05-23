import numpy as np
import numexpr as ne
import timeit
from math import sqrt, acos, cos
from matplotlib.path import Path
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment

a = np.arange(0, 30, 0.1)
b = np.arange(0, 30, 0.1)
X, Y = np.meshgrid(a, b)

W = np.vstack([X.ravel(), Y.ravel()])
W = W.transpose()

def npn():

	return np.linalg.norm(W)

def nen():

	x = W[:,0]; y = W[:,1]
	return ne.evaluate('sqrt(x**2 + y**2)')

def circumcenter(x1, y1, x2, y2, x3, y3):

	# d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
	# ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
	# uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

	# return ux, uy

	mid_ab_x = (x1 + x2) / 2
	mid_ab_y = (y1 + y2) / 2

	mid_bc_x = (x2 + x3) / 2
	mid_bc_y = (y2 + y3) / 2

	slope_ab = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
	slope_bc = (y3 - y2) / (x3 - x2) if x3 != x2 else float('inf')

	if slope_ab == 0:

		slope_bc = -float('inf')
	elif slope_bc == 0:

		slope_ab = -float('inf')
	else:

		slope_ab = -1 / slope_ab
		slope_bc = -1 / slope_bc

	center_x = (mid_bc_y - mid_ab_y + slope_ab * mid_ab_x - slope_bc * mid_bc_x) / (slope_ab - slope_bc)
	center_y = mid_ab_y + slope_ab * (center_x - mid_ab_x)

	radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5

	return center_x, center_y, radius

def calculate_tangent_angle(circle_center, circle_radius, point):

	distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)
	adjcent = np.sqrt(distance**2 - circle_radius**2)
	angle = 2*np.arctan(circle_radius/adjcent)

	# # Calculate the distance between the circle center and the point
	# distance = np.sqrt((point[0] - circle_center[0])**2 + (point[1] - circle_center[1])**2)

	# # Calculate the length of the line segment connecting the circle center and the point
	# length = distance - circle_radius

	# # Calculate the angle using the arctan function
	# angle = 2*np.arctan(length / (2*circle_radius))

	return (angle*180)/np.pi

if __name__ == '__main__':

	number = 1

	# x = timeit.timeit(stmt = "npn()", number = number, globals = globals())
	# print(x)
	# x = timeit.timeit(stmt = "nen()", number = number, globals = globals())
	# print(x)


	# cost_1 = np.array([[4, 1, 3], [10, 10, 10], [10, 10, 10]])
	# row_ind, col_ind = linear_sum_assignment(cost_1)

	# print(row_ind, col_ind)

	# for row, col in zip(row_ind, col_ind):

	# 	print(f"Agent {row+1} assigned to Task {col+1}")

	circle_center = (0, 0)
	circle_radius = 5
	point = (8, 0)

	tangent_angle = calculate_tangent_angle(circle_center, circle_radius, point)
	print("Tangent angle:", tangent_angle)