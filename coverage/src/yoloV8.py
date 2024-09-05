#!/usr/bin/env python3

import cv2
import torch
import rospy
import numpy as np
from time import time
from ultralytics import YOLO
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from coverage.msg import BoundingBox, BoundingBoxes

class Yolo_Detection:

	def __init__(self,):

		# load parameters
		self.image_topic = rospy.get_param('yoloV8_ros/image_topic')
		self.pub_topic = rospy.get_param('yoloV8_ros/pub_topic')
		self.output_topic = rospy.get_param('yoloV8_ros/output_topic')
		self.conf = rospy.get_param('yoloV8_ros/conf')
		self.visualize = rospy.get_param('yoloV8_ros/visualize')
		self.camera_frame = rospy.get_param('yoloV8_ros/camera_frame')
		self.sort = rospy.get_param('yoloV8_ros/sort')
		# print("image_topic: ", rospy.get_param('yoloV8_ros/image_topic'))

		# Which device to use
		self.device = 0

		# Model
		weight_path = rospy.get_param('yoloV8_ros/weight_path')
		self.model = YOLO(weight_path)
		self.model.fuse()

		self.model.conf = self.conf
		self.color_image = Image()
		self.getImageStatus = False

		# image subscribe
		self.color_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)

		# output publishers
		self.position_pub = rospy.Publisher(self.pub_topic, BoundingBoxes, queue_size=1)

		self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

		# if no image messages
		while (not self.getImageStatus):

			rospy.loginfo("waiting for image.")
			rospy.sleep(2)

	def image_callback(self, image):

		self.boundingBoxes = BoundingBoxes()
		self.boundingBoxes.header = image.header
		self.boundingBoxes.image_header = image.header
		self.getImageStatus = True
		self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)

		self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

		# results = self.model(self.color_image, show=False, conf=0.25, device=self.device)
		# results = self.model.track(source=self.color_image, show=False, conf=0.25, device=self.device, tracker=self.sort)
		results = self.model.track(source=self.color_image, show=False, conf=0.25, device=self.device, tracker=self.sort, stream=True)
		result_ = [result for result in results]

		self.dectshow(result_, image.height, image.width)

	def dectshow(self, results, height, width):

		self.frame = results[0].plot()
		print(str(results[0].speed['inference']))
		fps = 1000.0/ results[0].speed['inference']
		# cv2.putText(self.frame, f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

		boxes = results[0].boxes
		track_ids = results[0].boxes.id.int().cpu().tolist()

		# for result in results[0].boxes:
		for box, track_id in zip(boxes, track_ids):

			boundingBox = BoundingBox()
			boundingBox.id = track_id
			boundingBox.xmin = np.int64(box.xyxy[0][0].item())
			boundingBox.ymin = np.int64(box.xyxy[0][1].item())
			boundingBox.xmax = np.int64(box.xyxy[0][2].item())
			boundingBox.ymax = np.int64(box.xyxy[0][3].item())
			boundingBox.Class = results[0].names[box.cls.item()]
			boundingBox.probability = box.conf.item()

			self.boundingBoxes.bounding_boxes.append(boundingBox)
		self.position_pub.publish(self.boundingBoxes)
		self.publish_image(self.frame, height, width)

		# if self.visualize :

		# 	cv2.imshow('YOLOv8', self.frame)

	def publish_image(self, imgdata, height, width):

		image_temp = Image()
		header = Header(stamp=rospy.Time.now())
		header.frame_id = self.camera_frame
		image_temp.height = height
		image_temp.width = width
		image_temp.encoding = 'bgr8'
		image_temp.data = np.array(imgdata).tobytes()
		image_temp.header = header
		image_temp.step = width * 3

		self.image_pub.publish(image_temp)

if __name__ == "__main__":

	try:
		rospy.init_node('yolov8_ros', anonymous=True)
		yolo_dect = Yolo_Detection()

		while not rospy.is_shutdown():

			rate = rospy.Rate(100)
			rate.sleep()

	except rospy.ROSInterruptException:

		pass