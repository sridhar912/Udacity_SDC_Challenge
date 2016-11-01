
"""Routine for decoding the binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import tensorflow as tf

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 101267
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 3812
NUM_CLASS_LABELS = 100
STEERING_ANGLE_RANGE = 17 / NUM_CLASS_LABELS

def _generate_label(steering_angle):
    if tf.less(steering_angle, 0) is not None:
        label = tf.floor((NUM_CLASS_LABELS /2) + ((steering_angle / STEERING_ANGLE_RANGE)))
    else:
        label = tf.ceil((NUM_CLASS_LABELS / 2) - ((steering_angle / STEERING_ANGLE_RANGE)))

    return label


def read_and_decode_single_example(serialized_example):

  class readRecord(object):
    pass
  result = readRecord()

  features = tf.parse_single_example(
      serialized_example,
      features={
          # We know the length of both fields. If not the
          # tf.VarLenFeature could be used
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'image/steering_angle': tf.FixedLenFeature([1], dtype=tf.float32, default_value=0.0),
          'image/height': tf.FixedLenFeature([1], dtype=tf.int64, default_value=480),
          'image/width': tf.FixedLenFeature([1], dtype=tf.int64, default_value=640),
          'image/channels': tf.FixedLenFeature([1], dtype=tf.int64, default_value=3),
          'image/format': tf.FixedLenFeature([],dtype=tf.string,default_value='jpeg'),

      })
  # now return the converted data
  en_image = features['image/encoded']
  result.steering_angle = features['image/steering_angle']
  result.label = _generate_label(result.steering_angle)
  result.format = features['image/format']
  result.height = tf.cast(features['image/height'],dtype=tf.int32)
  result.width = tf.cast(features['image/width'],dtype=tf.int32)
  result.channels = tf.cast(features['image/channels'],dtype=tf.int32)
  if result.format == 'png':
      image = tf.image.decode_png(en_image)
  else:
      image = tf.image.decode_jpeg(en_image)

  # FIX_ME seems wierd, the size variable cant be a tensor..
  height = 480
  width = 640
  channels = 3
  image.set_shape([height, width, channels])

  result.image = image

  return result


def _generate_image_and_label_batch(image, steering_angle, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, steering_angle_batch = tf.train.shuffle_batch(
        [image, steering_angle],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, steering_angle_batch = tf.train.batch(
        [image, steering_angle],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.image_summary('images', images)

  return images, tf.reshape(steering_angle_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for training using the Reader ops.
  Args:
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [f for f in glob.glob(data_dir + '/*')]
  #filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Approximate number of examples per shard.
  examples_per_shard = 1024
  input_queue_memory_factor = 0.4
  num_readers = 4
  min_queue_examples = examples_per_shard * input_queue_memory_factor
  examples_queue = tf.RandomShuffleQueue(
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples,
          dtypes=[tf.string])

  # Create multiple readers to populate the queue of examples.
  if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
          reader = tf.TFRecordReader()
          _, value = reader.read(filename_queue)
          enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
  else:
      reader = tf.TFRecordReader()
      _, example_serialized = reader.read(filename_queue)

  images_and_lables_and_angles = []
  num_preprocess_threads = 16
  for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      read_input = read_and_decode_single_example(example_serialized)
      float_image = tf.image.convert_image_dtype(read_input.image, tf.float32)
      float_image = float_image / 255
      image_resize = tf.image.resize_images(float_image, [120, 160])
      #image_crop = tf.image.crop_to_bounding_box(image_resize, 60, 0, 60, 160)
      mean_image = tf.image.per_image_whitening(image_resize)
      #image_hsv = tf.image.rgb_to_hsv(image_crop)
      if not thread_id:
          tf.image_summary('original', tf.expand_dims(float_image,0))
          tf.image_summary('resize', tf.expand_dims(image_resize, 0))
          #tf.image_summary('crop', tf.expand_dims(image_crop, 0))
          #tf.image_summary('hsv', tf.expand_dims(image_resize, 0))
      images_and_lables_and_angles.append([mean_image, read_input.label, read_input.steering_angle])

  images_batch, labels_batch, angles_batch = tf.train.batch_join(
      images_and_lables_and_angles,
      batch_size=batch_size,
      capacity=2 * num_preprocess_threads * batch_size)

  tf.image_summary('images', images_batch)

  return images_batch, tf.reshape(labels_batch, [batch_size]), tf.reshape(angles_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
  """Construct input for evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [f for f in glob.glob(data_dir + '/*')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    filenames = [f for f in glob.glob(data_dir + '/*')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Approximate number of examples per shard.
  examples_per_shard = 1024
  input_queue_memory_factor = 0.4
  num_readers = 4
  min_queue_examples = examples_per_shard * input_queue_memory_factor
  examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size,
      dtypes=[tf.string])

  # Create multiple readers to populate the queue of examples.
  if num_readers > 1:
      enqueue_ops = []
      for _ in range(num_readers):
          reader = tf.TFRecordReader()
          _, value = reader.read(filename_queue)
          enqueue_ops.append(examples_queue.enqueue([value]))

      tf.train.queue_runner.add_queue_runner(
          tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
      example_serialized = examples_queue.dequeue()
  else:
      reader = tf.TFRecordReader()
      _, example_serialized = reader.read(filename_queue)

  images_and_lables_and_angles = []
  num_preprocess_threads = 16
  for thread_id in range(num_preprocess_threads):
      # Parse a serialized Example proto to extract the image and metadata.
      read_input = read_and_decode_single_example(example_serialized)
      float_image = tf.image.convert_image_dtype(read_input.image, tf.float32)
      image_resize = tf.image.resize_images(float_image, [120, 160])
      #image_crop = tf.image.crop_to_bounding_box(image_resize, 70, 0, 50, 160)
      mean_image = tf.image.per_image_whitening(image_resize)
      #image_hsv = tf.image.rgb_to_hsv(image_crop)
      images_and_lables_and_angles.append([mean_image,read_input.label ,read_input.steering_angle])

  images_batch, labels_batch, angles_batch = tf.train.batch_join(
      images_and_lables_and_angles,
      batch_size=batch_size,
      capacity=2 * num_preprocess_threads * batch_size)

  return images_batch, tf.reshape(labels_batch, [batch_size]), tf.reshape(angles_batch, [batch_size])
