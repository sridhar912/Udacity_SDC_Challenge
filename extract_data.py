# This code has been copied and modified from (https://github.com/DominicBreuker/self-driving-car-experiments#getting-data) 
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
images = []
tmp = []

image_timestamps = []

bridge = CvBridge()

def save_image(dir, msg, image_name):
    path_name = os.path.join(dir, image_name)
    try:
        cv2.imwrite(path_name, bridge.imgmsg_to_cv2(msg))
    except CvBridgeError as e:
        print(e)

#count used for image numbering
im_count = 1
#Use this variable to filter images which has not much steering movement.
angle_threshold = 0 #0.523599 # This value is equal to 30 degree. Set this value to zero, if you are want to save all the images
#Use average of angles until image timestamp
use_average = True

with rosbag.Bag(rosbag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == steering_report_topic:
            angle_values.append(msg.steering_wheel_angle)
            speed_values.append(msg.speed)
        elif topic == left_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            #save_image(left_images_dir, msg, image_name)
        elif topic == center_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            #save_image(center_images_dir, msg, image_name)
        elif topic == right_camera_topic:
            image_name = '{}{:08d}{}'.format("image_", im_count, ".png")
            #save_image(center_images_dir, msg, image_name)
            if not 0 in speed_values:
                if use_average:
                    angle_value = np.mean(angle_values)
                    if abs(angle_value) > angle_threshold:
                        print ""
                        image_names.append(image_name)
                        tmp.append(angle_value)
                        #save this image
                else:
                    # at any image timestamp, use the last known steering angle
                    angle_value = angle_values[-1]
                    if abs(angle_value) > angle_threshold:
                        print ""
                        image_names.append(image_name)
                        tmp.append(angle_value)
                        # save this image

            # reset the list to empty. Cant think of a better way to do it
            angle_values[:] = []
            speed_values[:] = []
            images[:] = []
            im_count = im_count + 1
            if im_count % 1000 == 0:
                print 'Done processing images :{}'.format(im_count)

steering_angle_file = os.path.join(data_dir, "data_new_1.txt")

with open(steering_angle_file, "w") as data_file:
    for idx in range(len(tmp)):
        line = '{},{}\n'.format(image_names[idx], tmp[idx])
        data_file.write(line)
