import numpy as np

if __name__ == '__main__':

	Rx = np.array([[1, 0, 0],
				[0, np.cos(-0.5*np.pi), -np.sin(-0.5*np.pi)],
				[0, np.sin(-0.5*np.pi), +np.cos(-0.5*np.pi)]])
	# Rxinv = np.array([[1, 0, 0],
	# 			[0, np.cos(0.5*np.pi), -np.sin(0.5*np.pi)],
	# 			[0, np.sin(0.5*np.pi), +np.cos(0.5*np.pi)]])
	Rxinv = np.linalg.inv(Rx)

	img0 = np.array([0, 1, 0])
	tau0 = np.matmul(Rxinv, np.matmul(Rx, np.reshape(img0, (3,1))))

	print("tau0: ", tau0)