import numpy as np
from math import sqrt, acos, cos
from matplotlib.path import Path
from shapely.geometry import Point
from scipy.stats import multivariate_normal
from shapely.geometry.polygon import Polygon
from scipy.optimize import linear_sum_assignment

if __name__ == '__main__':

	a = np.array([1,2,3,Nan])