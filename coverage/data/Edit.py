import os
import cv2

# Specify your save path
save_path = '/home/leo/mts/src/coverage/data/image/'

# Open the video file
cap = cv2.VideoCapture('drone_4.avi')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Total frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total_frames: ", total_frames)

# Determine the step size for 100 images
step_size = total_frames // 25

count = 0
img_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # If frame is read correctly, ret is True
        if count % step_size == 0 and img_count < 30:
            # Include the save path in the filename
            filename = os.path.join(save_path, f'drone_number{img_count+377}.jpg')
            cv2.imwrite(filename, frame)
            img_count += 1
        count += 1
    else:
        break

# Release the video capture object
cap.release()