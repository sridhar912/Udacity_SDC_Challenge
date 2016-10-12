import os
import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

data_dir = "../Dataset/"
rosbag_file = os.path.join(data_dir, "dataset.bag")

def get_image_dir(image_type):
    images_dir = os.path.join(data_dir, image_type)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

left_images_dir = get_image_dir("left")
center_images_dir = get_image_dir("center")
right_images_dir = get_image_dir("right")


steering_report_topic = "/vehicle/steering_report"
left_camera_topic = "/left_camera/image_color"
center_camera_topic = "/center_camera/image_color"
right_camera_topic = "/right_camera/image_color"
topics = [steering_report_topic, left_camera_topic,
          center_camera_topic, right_camera_topic]

angle_timestamps = []
angle_values = []
speed_values = []
image_names = []

image_timestamps = []

bridge = CvBridge()

def save_image(dir, msg, image_name):
    path_name = os.path.join(dir, image_name)
    try:
        cv2.imwrite(path_name, bridge.imgmsg_to_cv2(msg))
    except CvBridgeError as e:
        print(e)

im_count = 1

with rosbag.Bag(rosbag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == steering_report_topic:
            angle_timestamps.append(msg.header.stamp.to_nsec())
            angle_values.append(msg.steering_wheel_angle)
            speed_values.append(msg.speed)
        elif topic == left_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            save_image(left_images_dir, msg, image_name)
        elif topic == center_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            save_image(center_images_dir, msg, image_name)
        elif topic == right_camera_topic:
            image_timestamps.append(msg.header.stamp.to_nsec())
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            save_image(right_images_dir, msg, image_name)
            image_names.append(image_name)
            im_count = im_count + 1
            if im_count % 1000 == 0:
                print 'Done processing images :{}'.format(im_count)


#The following code will generate a text file. This file can be used to read the images saved in the dir mentioned above
angles_at_timestamps = []
image_at_timestape = []

#Use this variable to filter images which has not much steering movement.
angle_threshold = 0.523599 # This value is equal to 30 degree. Set this value to zero, if you are want to use all the angles

def get_angles_at_timestamps(use_average = True):
    angle_idx = 0
    for image_idx in range(len(image_timestamps)):
        # go through angle values until we reach current image time
        angles_until_timestamps = []
        speed_until_timestamps = []
        while angle_idx < len(angle_timestamps) and angle_timestamps[angle_idx] <= image_timestamps[image_idx]:
            angles_until_timestamps.append(angle_values[angle_idx])
            speed_until_timestamps.append(speed_values[angle_idx])
            angle_idx += 1
        avg_angle = np.average(angles_until_timestamps)
        # If the speed is zero, skip the image. Also skip if average angle is less that angle threshold
        #Comment the below line, if you need to use all the images for processing
        if not 0 in speed_until_timestamps and abs(avg_angle) >= angle_threshold:
            if use_average:
                # at any image timestamp, use the average value of the steering from last known timestamp
                angles_at_timestamps.append(avg_angle)
            else:
                # at any image timestamp, use the last known steering angle
                angles_at_timestamps.append(angle_values[angle_idx - 1])
            image_at_timestape.append(image_names[image_idx])

get_angles_at_timestamps(True)

steering_angle_file = os.path.join(data_dir, "data.txt")

with open(steering_angle_file, "w") as data_file:
    for idx in range(len(angles_at_timestamps)):
        line = '{},{}\n'.format(image_at_timestape[idx], angles_at_timestamps[idx])
        data_file.write(line)
