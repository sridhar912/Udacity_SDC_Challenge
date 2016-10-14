# This code has been copied and modified from this github repo https://github.com/rwightman/udacity-driving-reader
# The intension is to completely understand the code and use it as required by me
import os
import rosbag
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError

data_dir = "../Dataset/"
rosbag_file = os.path.join(data_dir, "dataset.bag")

def get_tf_dir(image_type):
    images_dir = os.path.join(data_dir, image_type)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

tf_dir = get_tf_dir("tfRecord")

steering_report_topic = "/vehicle/steering_report"
left_camera_topic = "/left_camera/image_color"
center_camera_topic = "/center_camera/image_color"
right_camera_topic = "/right_camera/image_color"
topics = [steering_report_topic, left_camera_topic, center_camera_topic, right_camera_topic]
camera_topics = [left_camera_topic, center_camera_topic, right_camera_topic]
tf_topics = [center_camera_topic]

angle_values = []
speed_values = []
image_names = []
image_timestamps = []
images = []
tmp = []

bridge = CvBridge()

#count used for image numbering
im_count = 1
#Use this variable to filter images which has not much steering movement.
angle_threshold = 0.523599 # This value is equal to 30 degree. Set this value to zero, if you are want to save all the images
#Use average of angles until image timestamp
use_average = True


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def store_msg(tfRecord, msg_angle, msg_image, img_fmt = 'jpeg'):
    cv_image = bridge.imgmsg_to_cv2(msg_image, "bgr8")
    _, encoded = cv2.imencode('.' + img_fmt, cv_image)
    colorspace = b'RGB'
    channels = 3
    feature_dict = {
        'image/steering_angle': _float_feature(msg_angle),
        'image/height': _int64_feature(msg_image.height),
        'image/width': _int64_feature(msg_image.width),
        'image/channels': _int64_feature(channels),
        'image/colorspace': _bytes_feature(colorspace),
        'image/format': _bytes_feature(img_fmt),
        'image/encoded': _bytes_feature(encoded.tobytes())
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    tfRecord.add_example(example)


class cTfWriter():
    #TF reader is used to load the images in faster way using queue mechanism rather than doing old style loop read
    #Good tutorial is availble here.. https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    #num_image -- total number of images in the rosbag
    #num_bins  -- images are stored in .bin format. This gives numbers of bins to be used for converting num_images equally (almost)
    #images_per_bin -- number of images to be stored in a single .bin file
    def __init__(self,outdir,name,num_images,num_bins = 128):
        self.outdir = outdir
        self.name = name
        self.num_images = num_images
        self.num_bins = num_bins
        self.images_per_bin = num_images // num_bins
        self.tfWriter = None
        self.per_bin_count = 0
        self.total_count = 0

    def create_tfWriter(self):
        if not self.tfWriter or self.per_bin_count >= self.images_per_bin:
            bin_number = self.total_count // self.images_per_bin
            assert (bin_number <= self.num_bins)
            output_filename = '%s-%.5d-of-%.5d' % (self.name, bin_number, self.num_bins)
            output_file = os.path.join(self.outdir, output_filename)
            self.tfWriter = tf.python_io.TFRecordWriter(output_file)
            self.per_bin_count = 0

    def add_example(self,example):
        self.create_tfWriter()
        self.tfWriter.write(example.SerializeToString())
        self.per_bin_count += 1
        self.total_count += 1
        if not self.total_count % 1000:
            print('Processed %d of %d images for %s' % (self.total_count, self.num_images, self.name))
            sys.stdout.flush()


def calc_angle(angle_values, speed_values):
    # If the speed is zero, skip the image.
    # Comment the below line, if you need to use all the images for processing
    if not 0 in speed_values:
        if use_average:
            angle_value = np.mean(angle_values)
            # Also skip if average angle is less that angle threshold
	    # Comment the below line, if you need to use all the images for processing
            if abs(angle_value) > angle_threshold:
                return True, angle_value
            else:
                return False, angle_value
        else:
            # at any image timestamp, use the last known steering angle
            angle_value = angle_values[-1]
            if abs(angle_value) > angle_threshold:
                return True, angle_value
            else:
                return False, angle_value

# first dataset contains 15212 images.. Here validation set is not created.
tfRecord = cTfWriter(tf_dir,'train',15212,10)

with rosbag.Bag(rosbag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == steering_report_topic:
            angle_values.append(msg.steering_wheel_angle)
            speed_values.append(msg.speed)
        elif topic in camera_topics:
            if topic == left_camera_topic:
                record, angle_value = calc_angle(angle_values,speed_values)
                if record and topic in tf_topics:
                    store_msg(tfRecord,angle_value,msg)
            elif topic == center_camera_topic:
                record, angle_value = calc_angle(angle_values, speed_values)
                if record and topic in tf_topics:
                    store_msg(tfRecord, angle_value, msg)
            elif topic == right_camera_topic:
                record, angle_value = calc_angle(angle_values, speed_values)
                if record and topic in tf_topics:
                    store_msg(tfRecord, angle_value, msg)
                # reset the list to empty. Cant think of a better way to do it
                angle_values[:] = []
                speed_values[:] = []
                images[:] = []
                im_count = im_count + 1
                if im_count % 1000 == 0:
                    print 'Done processing images :{}'.format(im_count)

#Check the code by extracting the information..
#This is a simple check to display the values
check_angle = True
if check_angle:
    filename = '%s/%s-%.5d-of-%.5d' % (tf_dir,'train', 0 , 10)  # supply sample values
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        # traverse the Example format to get data
        image = example.features.feature['image/encoded'].bytes_list.value
        steering_angle = example.features.feature['image/steering_angle'].float_list.value[0]
        print 'The steering angle {}'.format(steering_angle)

check_image = True
if check_image:
    def read_and_decode_single_example(filename):
        # first construct a queue containing a list of filenames.
        # this lets a user split up there dataset in multiple files to keep
        # size down
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=None)
        # Unlike the TFRecordWriter, the TFRecordReader is symbolic
        reader = tf.TFRecordReader()
        # One can read a single serialized example from a filename
        # serialized_example is a Tensor of type string.
        _, serialized_example = reader.read(filename_queue)
        # The serialized example is converted back to actual values.
        # One needs to describe the format of the objects to be returned
        features = tf.parse_single_example(
            serialized_example,
            features={
                # We know the length of both fields. If not the
                # tf.VarLenFeature could be used
                'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
                'image/steering_angle': tf.FixedLenFeature([1], dtype=tf.float32, default_value=0.0),
            })
        # now return the converted data
        image = features['image/encoded']
        steering_angle = features['image/steering_angle']
        return steering_angle, image


    # returns symbolic label and image
    filename = '%s/%s-%.5d-of-%.5d' % (tf_dir, 'train', 0, 10)
    steering_angle, image = read_and_decode_single_example(filename)
    image = tf.image.decode_jpeg(image)

    sess = tf.Session()

    # Required. See below for explanation
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    # grab examples back.
    # first example from file
    steering_angle_1, image_val_1 = sess.run([steering_angle, image])
    # second example from file
    steering_angle_2, image_val_2 = sess.run([steering_angle, image])

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(image_val_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_val_2)
    plt.show()
    print "Image checking done.."
