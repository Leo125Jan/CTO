import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import csv
import pygame
import random
import numpy as np
from PTZCAM import PTZcon
from time import sleep, time
from math import cos, acos, sqrt, exp, sin
from scipy.stats import multivariate_normal

initialized = False

class UAVs():

    def __init__(self, map_size, resolution):
        
        self.members = []
        self.map_size = map_size
        self.grid_size = resolution

    def AddMember(self, ptz_info):
        
        ptz = PTZcon(ptz_info, self.map_size, self.grid_size)
        self.members.append(ptz)
        
        return

    # inefficient way, might come up with other data structure to manage the swarm 
    def DeleteMember(self, id): 
        
        for i in range(len(self.members)):
            if self.members.id == id:
                del self.members[i]
                break
        return

class Visualize():

	def __init__(self, map_size, grid_size):

		self.size = (np.array(map_size)/np.array(grid_size)).astype(np.int64)
		self.grid_size = grid_size
		self.window_size = np.array(self.size)*4
		self.display = pygame.display.set_mode(self.window_size)
		self.display.fill((0,0,0))
		self.blockSize = int(self.window_size[0]/self.size[0]) #Set the size of the grid block

		for x in range(0, self.window_size[0], self.blockSize):

		    for y in range(0, self.window_size[1], self.blockSize):

		        rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
		        pygame.draw.rect(self.display, (125,125,125), rect, 1)

		pygame.display.update()
	
	def Visualize2D(self, cameras, event_plt, targets, circumcenter_center, circumcenter_radius, \
					side_center, side_center_radius):

		map_plt = np.zeros(np.shape(event_plt)[0]) - 1

		for i in range(len(cameras)):

			if i == 1:

				for j in range(np.shape(event_plt)[0]):

					if map_plt[j] == 0 and cameras[i].map_plt[j] > 0:

						cameras[i].map_plt[j] = 1

			map_plt = cameras[i].map_plt + map_plt

		count = 0
		for y in range(0, self.window_size[0], self.blockSize):

			for x in range(0, self.window_size[1], self.blockSize):

				dense = event_plt[count]
				w = 0.6
				id = int(map_plt[count])

				if id == -1:
					gray = (1-w)*125 + w*dense
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, (gray, gray, gray), rect, 0)
				elif id not in range(len(cameras)):

					if id == 3:# N: id-2 -> Green of head of blue
						color = ((1-w)*cameras[id-1].color[0] + w*dense,\
								(1-w)*cameras[id-1].color[1] + w*dense,\
								(1-w)*cameras[id-1].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 4:
						color = ((1-w)*cameras[id-4].color[0] + w*dense,\
								(1-w)*cameras[id-4].color[1] + w*dense,\
								(1-w)*cameras[id-4].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
					elif id == 5:
						color = ((1-w)*cameras[id-5].color[0] + w*dense,\
								(1-w)*cameras[id-5].color[1] + w*dense,\
								(1-w)*cameras[id-5].color[2] + w*dense)
						rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
						pygame.draw.rect(self.display, color, rect, 0)
				else:
					color = ((1-w)*cameras[id].color[0] + w*dense,\
							(1-w)*cameras[id].color[1] + w*dense,\
							(1-w)*cameras[id].color[2] + w*dense)
					rect = pygame.Rect(x, y, self.blockSize, self.blockSize)
					pygame.draw.rect(self.display, color, rect, 0)

				count += 1

		for camera in cameras:

			color = (camera.color[0], camera.color[1], camera.color[2])
			center = camera.pos/self.grid_size*self.blockSize

			R = camera.R*cos(camera.alpha)/self.grid_size[0]*self.blockSize
			pygame.draw.line(self.display, color, center, center + camera.perspective*R, 3)
			pygame.draw.circle(self.display, color, camera.pos/self.grid_size*self.blockSize, 10)

		for camera in cameras:

			color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
			pygame.draw.polygon(self.display, color, [camera.pos/self.grid_size*self.blockSize, \
			                                            camera.ltop/self.grid_size*self.blockSize, \
			                                            camera.top/self.grid_size*self.blockSize, \
			                                            camera.rtop/self.grid_size*self.blockSize], 2)

		for target in targets:

			pygame.draw.circle(self.display, (0,0,0), np.asarray(target[0])/self.grid_size\
			                    *self.blockSize, 6)

		# for (center, r) in zip(side_center, side_center_radius):

		# 	pygame.draw.circle(self.display, (129, 128, 157), center/self.grid_size\
		#                     *self.blockSize, r*(40), 2)

		# for camera in cameras:

		# 	color = (camera.color[0]*0.5, camera.color[1]*0.5, camera.color[2]*0.5)
		# 	pygame.draw.circle(self.display, color, camera.incircle[0]/self.grid_size\
		#                     *self.blockSize, camera.incircle[1]*(40), 5)

		# pygame.draw.circle(self.display, (183, 158, 158), circumcenter_center/self.grid_size\
		#                     *self.blockSize, circumcenter_radius*(40), 2)

		for camera in cameras:

			color = (camera.color[0]*0.7, camera.color[1]*0.7, camera.color[2]*0.7)
			pygame.draw.circle(self.display, color, np.asarray(camera.target[0][0])/self.grid_size\
			                    *self.blockSize, 3)

		pygame.draw.rect(self.display, (0, 0, 0), (0, 0, map_size[0]/grid_size[0]*self.blockSize, \
                                                        map_size[1]/grid_size[1]*self.blockSize), width = 3)
		pygame.display.flip()

		# print(halt)
	
def norm(arr):

	sum = 0

	for i in range(len(arr)):

	    sum += arr[i]**2

	return sqrt(sum)

def event_density(event, targets, W):

	for target in targets:

		F = multivariate_normal([target[0][0], target[0][1]],\
						[[target[1], 0.0], [0.0, target[1]]])
		
		event += F.pdf(W)

	return 0 + event

def TargetDynamic(x, y):

	dx = np.random.uniform(-0.5, 0.5, 1)
	dy = np.random.uniform(-0.5, 0.5, 1)

	return (x, y)
	#(np.round(float(np.clip(dx/2 + x, 0, 24)),1), np.round(float(np.clip(dy/2 + y, 0, 24)),1))

def circumcenter(targets):

	for i in range(0, len(targets)):

		globals()["x" + str(i+1)] = targets[i][0][0]
		globals()["y" + str(i+1)] = targets[i][0][1]

	d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
	center_x = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
	center_y = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d

	radius = ((center_x - x1) ** 2 + (center_y - y1) ** 2) ** 0.5

	return (center_x, center_y), radius

def sidecenter(targets):

	for i in range(0, len(targets)):

		globals()["x" + str(i+1)] = targets[i][0][0]
		globals()["y" + str(i+1)] = targets[i][0][1]

	side_center_1 = np.array([0.5*(x1 + x2), 0.5*(y1 + y2)]); radius_1 = 0.5*np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
	side_center_2 = np.array([0.5*(x1 + x3), 0.5*(y1 + y3)]); radius_2 = 0.5*np.sqrt( (x1-x3)**2 + (y1-y3)**2 )
	side_center_3 = np.array([0.5*(x2 + x3), 0.5*(y2 + y3)]); radius_3 = 0.5*np.sqrt( (x2-x3)**2 + (y2-y3)**2 )

	side_center = [side_center_1, side_center_2, side_center_3]
	side_center_radius = [radius_1, radius_2, radius_3]

	return side_center, side_center_radius

if __name__ == "__main__":

	pygame.init()

	map_size = np.array([25, 25])
	grid_size = np.array([0.1, 0.1])

	cameras = []
	cameras_pos = []

	camera0 = { 'id'            :  0,
				'position'      :  np.array([2.0, 2.0]),
				'perspective'   :  np.array([0.5, 0.5]),
				'AngleofView'   :  20,
				'range_limit'   :  4,
				'lambda'        :  2,
				'color'         : (200, 0, 0)}
	cameras.append(camera0)

	camera1 = { 'id'            :  1,
				'position'      :  np.array([23.0, 2.0]),
				'perspective'   :  np.array([1.0, 0.0]),
				'AngleofView'   :  20,
				'range_limit'   :  4,
				'lambda'        :  2,
				'color'         : (0, 200, 0)}
	cameras.append(camera1)

	camera2 = { 'id'            :  2,
				'position'      :  np.array([2.0, 23.0]),
				'perspective'   :  np.array([1.0, 0.0]),
				'AngleofView'   :  20,
				'range_limit'   :  4,
				'lambda'        :  2,
				'color'         : (0, 0, 200)}
	cameras.append(camera2)

	# camera3 = { 'id'            :  3,
	# 			'position'      :  np.array([23.0, 23.0]),
	# 			'perspective'   :  np.array([1.0, 0.0]),
	# 			'AngleofView'   :  20,
	# 			'range_limit'   :  4,
	# 			'lambda'        :  2,
	# 			'color'         : (255, 150, 0)}
	# cameras.append(camera3)
	#
	# camera4 = { 'id'            :  4,
	# 			'position'      :  np.array([12.5, 2.0]),
	# 			'perspective'   :  np.array([1.0, 0.0]),
	# 			'AngleofView'   :  20,
	# 			'range_limit'   :  4,
	# 			'lambda'        :  2,
	# 			'color'         : (255, 250, 0)}
	# cameras.append(camera4)

	# camera5 = { 'id'            :  5,
	# 			'position'      :  np.array([23.0, 12.5]),
	# 			'perspective'   :  np.array([0.5, 0.5]),
	# 			'AngleofView'   :  20,
	# 			'range_limit'   :  4,
	# 			'lambda'        :  2,
	# 			'color'         : (0, 240, 255)}
	# cameras.append(camera5)

	# camera6 = { 'id'            :  6,
	# 			'position'      :  np.array([12.5, 23.0]),
	# 			'perspective'   :  np.array([1.0, 0.0]),
	# 			'AngleofView'   :  20,
	# 			'range_limit'   :  4,
	# 			'lambda'        :  2,
	# 			'color'         : (150, 0, 255)}
	# cameras.append(camera6)

	# camera7 = { 'id'            :  7,
	# 			'position'      :  np.array([2.0, 12.5]),
	# 			'perspective'   :  np.array([1.0, 0.0]),
	# 			'AngleofView'   :  20,
	# 			'range_limit'   :  4,
	# 			'lambda'        :  2,
	# 			'color'         : (255, 0, 250)}
	# cameras.append(camera7)

	# for i in range(len(cameras)):

	# 	# filename = "D:/上課資料/IME/實驗室研究/Paper/Coverage Control/Quality based switch mode/Data/"
	# 	filename = "/home/leo/mts/src/QBSM/Data/Cost/"
	# 	# filename += "Data_" + str(i) + ".csv"
	# 	filename += "Cost_" + str(i) + ".csv"

	# 	f = open(filename, "w+")
	# 	f.close()

	# Initialize UAV team with PTZ cameras
	uav_team = UAVs(map_size, grid_size)

	for camera in cameras:
		uav_team.AddMember(camera)
		cameras_pos.append(camera["position"])

	# initialize environment with targets
	size = (map_size/grid_size).astype(np.int64)
	x_range = np.arange(0, map_size[0], grid_size[0])
	y_range = np.arange(0, map_size[1], grid_size[1])
	X, Y = np.meshgrid(x_range, y_range)

	W = np.vstack([X.ravel(), Y.ravel()])
	W = W.transpose()

	# target's [position, certainty, weight, velocity]
	targets = [[(6.5, 19), 1, 10], [(6.0, 18.0), 1, 10], [(7.0, 18.0), 1, 10]]
	# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10]]
	# targets = [[(12.0, 12.0), 1, 10], [(12.0, 13.0), 1, 10], [(13.0, 12.0), 1, 10], [(13.0, 13.0), 1, 10],
				# [(10.5, 12.5), 1, 10], [(12.5, 9.5), 1, 10], [(16.5, 12.5), 1, 10], [(12.5, 17.5), 1, 10]]
	velocities = np.random.rand(len(targets), 2) - 0.5  # Random initial velocities (-0.5 to 0.5)

	# Start Simulation
	Done = False
	vis = Visualize(map_size, grid_size)
	last = time()

	while not Done:

		for op in pygame.event.get():

			if op.type == pygame.QUIT:

				Done = True

		# if np.round(time() - last, 2) > 30.00 and np.round(time() - last, 2) < 50.00:

		# 	targets[0][0] = (targets[0][0][0] + 0.00, targets[0][0][1] + 0.005)
		# 	targets[1][0] = (targets[1][0][0] - 0.005, targets[1][0][1] - 0.02)
		# 	targets[2][0] = (targets[2][0][0] + 0.03, targets[2][0][1] - 0.04)

		# 	sleep(0.001)
		# elif np.round(time() - last, 2) > 50.00 and np.round(time() - last, 2) < 80.00:

		# 	targets[0][0] = (targets[0][0][0] + 0.03, targets[0][0][1] - 0.01)
		# 	targets[1][0] = (targets[1][0][0] + 0.01, targets[1][0][1] - 0.025)
		# 	targets[2][0] = (targets[2][0][0] + 0.01, targets[2][0][1] - 0.01)

		# 	sleep(0.001)

		# elif np.round(time() - last, 2) > 80.00 and np.round(time() - last, 2) < 110.00:

		# 	targets[0][0] = (targets[0][0][0] + 0.007, targets[0][0][1] - 0.035)
		# 	targets[1][0] = (targets[1][0][0] + 0.0292, targets[1][0][1] - 0.0022)
		# 	targets[2][0] = (targets[2][0][0] + 0.005, targets[2][0][1] - 0.005)

		# 	sleep(0.001)

		if np.round(time() - last, 2) > 20.00 and np.round(time() - last, 2) < 110.00:

			# Simulation parameters
			time_step = 0.1  # Time step in seconds
			min_distance = 0.7  # Minimum distance between points to avoid collision
			boundary_margin = 3  # Minimum distance from the boundary
			tracker_margin = 1  # Minimum distance from the boundary

			# Initialize point positions and velocities
			positions = [target[0] for target in targets]
			# print("positions: " + str(positions))
			# velocities = np.random.rand(4, 2) - 0.5  # Random initial velocities (-0.5 to 0.5)
			# print("velocities: " + str(velocities))

			positions += velocities * time_step
			# print("positions: " + str(positions))

			# Change direction of velocities every 3 seconds
			if np.round(time()-last, 0)%5 == 0:

				velocities = np.random.rand(len(targets), 2) - 0.5

			# Check for collisions and adjust velocities if necessary
			for i in range(len(positions)):

				# for j in range(i + 1, 4):
				for j in range(len(positions)):

					if j != i:
					
						dist = np.linalg.norm(positions[i] - positions[j])

						if dist < min_distance:

							# Adjust velocities to avoid collision
							direction = positions[i] - positions[j]
							velocities[i] = +(direction/np.linalg.norm(direction))*0.8
							velocities[j] = -(direction/np.linalg.norm(direction))*0.8

				# for  in range(len(positions)):

				if abs(positions[i, 0] - 0) <= boundary_margin or abs(positions[i, 0] - 25) <= boundary_margin:

					velocities[i, 0] *= -1  # Reverse x-direction velocity
				if abs(positions[i, 1] - 0) <= boundary_margin or abs(positions[i, 1] - 25) <= boundary_margin:

					velocities[i, 1] *= -1  # Reverse y-direction velocity

				for k in range(len(positions)):

					dist = np.linalg.norm(positions[i] - cameras_pos[k])

					if dist < tracker_margin:

						# Adjust velocities to avoid collision
						direction = positions[i] - cameras_pos[k]
						velocities[i] = +(direction/np.linalg.norm(direction))*0.8

			for (i, element) in zip(range(len(positions)), positions):

				targets[i][0] = element

		event = np.zeros(np.shape(W)[0])
		event1 = event_density(event, targets, W)
		event_plt1 = ((event - event1.min()) * (1/(event1.max() - event1.min()) * 255)).astype('uint8')

		past = time()
		for i in range(len(uav_team.members)):

			neighbors = [uav_team.members[j] for j in range(len(uav_team.members)) if j != i]
			uav_team.members[i].UpdateState(targets, neighbors, np.round(time() - last, 2))
		print("Simulation Time: " + str(time() - last))
		print("Calculation Time: " + str(time() - past), "\n")

		circumcenter_center, circumcenter_radius = circumcenter(targets)
		side_center, side_center_radius = sidecenter(targets)

		vis.Visualize2D(uav_team.members, event_plt1, targets, circumcenter_center, circumcenter_radius, \
						side_center, side_center_radius)

		if np.round(time() - last, 2) > 50.00:

			sys.exit()

	pygame.quit()