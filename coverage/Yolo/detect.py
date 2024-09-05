import wandb
import torch
from PIL import Image
from ultralytics import YOLO
from roboflow import Roboflow

wandb.init(project="yolov8")

if __name__ == '__main__':

	train = False
	predict = True

	if torch.cuda.is_available() and torch.cuda.device_count() and train:

		rf = Roboflow(api_key="p4hd1GJrlac9Sf4rOSn0")
		project = rf.workspace("drone-detection-u4baf").project("solo-detection")
		version = project.version(4)
		dataset = version.download("yolov8")

		model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
		
		# Train the model
		results = model.train(data='/home/leo/mts/src/coverage/Yolo/Solo Detection.v4i.yolov8/data.yaml', epochs=100, imgsz=640, device=0)

	if predict:

		model = YOLO("/home/leo/mts/src/coverage/Yolo/runs/detect/train8/weights/best.pt")

		source = "/home/leo/mts/src/coverage/Yolo/drone-detection-2-1/left0000.jpg"

		results1 = model.predict(source=source, save=True, imgsz=640, conf=0.25)
		# results = model.track(source=source, tracker="bytetrack.yaml")
		print("results1: ", results1)
		print("results1[0]: ", results1[0])
		print("frame: ", results1[0].plot())

		# for result in results1:

		# 	boxes = result.boxes  # Boxes object for bounding box outputs
		# 	masks = result.masks  # Masks object for segmentation masks outputs
		# 	keypoints = result.keypoints  # Keypoints object for pose outputs
		# 	probs = result.probs  # Probs object for classification outputs
		# 	print("boxes: ", boxes)
		# 	print("id: ", boxes.id)
		# 	result.show()  # display to screen
		# 	result.save(filename='/home/leo/mts/src/coverage/Yolo/drone-detection-2-1/result.jpg')  # save to disk

		results2 = model.predict(source=source, save=True, imgsz=640, conf=0.25, stream=True)
		result_ = [result for result in results2]
		print("results2: ", result_)
		print("results2[0]: ", result_[0])
		print("frame: ", result_[0].plot())