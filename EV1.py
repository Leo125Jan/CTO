import numpy as np
from math import sqrt, acos, cos
from matplotlib.path import Path
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':

	pos = np.array([0,0])
	a = np.array([-2,5])
	b = np.array([2,5])

	perspective = np.array([0,1])
	v_l = (a - pos)/np.linalg.norm(a - pos)
	v_r = (b - pos)/np.linalg.norm(b - pos)
	print(np.cross(v_l, perspective))
	print(np.cross(v_r, perspective))
