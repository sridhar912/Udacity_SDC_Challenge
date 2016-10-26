"""Builds the network for steering predict.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference_nvidia(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

import tensorflow as tf
from pip._vendor.distlib._backport.tarfile import TruncatedHeaderError

import predict_steering_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size_flag', 256,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir_flag', '../dataset',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16_flag', False,
                            """Train the model using fp16.""")

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = predict_steering_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = predict_steering_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16_flag else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, use_xavier, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16_flag else tf.float32
  if use_xavier:
      var = _variable_on_cpu(
          name,
          shape,
          tf.contrib.layers.xavier_initializer_conv2d())
  else:
      var = _variable_on_cpu(
          name,
          shape,
          tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir_flag:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir_flag, 'data_train')
  images, steering_angle = predict_steering_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size_flag)
  if FLAGS.use_fp16_flag:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(steering_angle, tf.float16)
  return images, steering_angle


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir_flag:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir_flag, 'data_eval')
  images, steering_angle = predict_steering_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size_flag)
  if FLAGS.use_fp16_flag:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(steering_angle, tf.float16)
  return images, steering_angle


def inference_nvidia(images):
  """Build the model as in nvidia paper of end to end learing.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  use_xavier_conv = True
  use_xavier = False
  keep_prob = 0.8
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         use_xavier_conv,
                                         shape=[5, 5, 3, 24],
                                         stddev=0.1,
                                         wd=0.00004)
    #conv = conv2d(images, kernel, 2)
    conv = tf.nn.conv2d(images, kernel, strides=[1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [24], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.elu(bias, name=scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             use_xavier_conv,
                                         shape=[5, 5, 24, 36],
                                         stddev=0.01,
                                         wd=0.00004)
        #conv = conv2d(conv1, kernel, 2)
        conv = tf.nn.conv2d(conv1, kernel, strides=[1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [36], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.elu(bias, name=scope.name)
        _activation_summary(conv2)

   # conv3
  with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             use_xavier_conv,
                                             shape=[5, 5, 36, 64],
                                             stddev=0.1,
                                             wd=0.00004)
        #conv = conv2d(conv2, kernel, 2)
        conv = tf.nn.conv2d(conv2, kernel, strides=[1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.elu(bias, name=scope.name)
        _activation_summary(conv3)

  # conv4
  with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             use_xavier_conv,
                                             shape=[3, 3, 64, 64],
                                             stddev=0.01,
                                             wd=0.00004)
        #conv = conv2d(conv3, kernel, 1)
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 2, 2, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.elu(bias, name=scope.name)
        _activation_summary(conv4)

  # fc1
  with tf.variable_scope('fc1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[512,384],
                                           stddev=0.01,
                                           wd=0.00004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        conv5_flat = tf.reshape(conv4, [-1, 512])
        fc1 = tf.nn.elu(tf.matmul(conv5_flat, kernel) + biases)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)
        _activation_summary(fc1_drop)

  # fc2
  with tf.variable_scope('fc2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[384,256],
                                           stddev=0.1,
                                           wd=0.00004)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.01))
        fc2 = tf.nn.elu(tf.matmul(fc1_drop, kernel) + biases)
        fc2_drop = tf.nn.dropout(fc2, keep_prob)
        _activation_summary(fc2_drop)

  # fc3
  with tf.variable_scope('fc3') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[256,50],
                                           stddev=0.05,
                                           wd=0.00004)
      biases = _variable_on_cpu('biases', [50], tf.constant_initializer(0.05))
      fc3 = tf.nn.elu(tf.matmul(fc2_drop, kernel) + biases)
      fc3_drop = tf.nn.dropout(fc3, keep_prob)
      _activation_summary(fc3_drop)

  # fc4
  with tf.variable_scope('fc4') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[50,10],
                                           stddev=0.01,
                                           wd=0.00004)
      biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.1))
      fc4 = tf.nn.elu(tf.matmul(fc3_drop, kernel) + biases)
      fc4_drop = tf.nn.dropout(fc4, keep_prob)
      _activation_summary(fc4_drop)

  # output
  with tf.variable_scope('output') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[10,1],
                                           stddev=0.1,
                                           wd=0.00004)
      biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.01))
      y = tf.mul(tf.atan(tf.matmul(fc4_drop, kernel) + biases), 5)
      #y = tf.nn.elu(tf.matmul(fc4_drop, kernel) + biases)
      #y = tf.matmul(fc4_drop, kernel) + biases
      _activation_summary(y)

  return y

# Implementation of VGG network
def get_conv2d(name, bottom, use_xavier, shape, stddev, wd, stride, bConst,  padding):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             use_xavier,
                                             shape=shape,
                                             stddev=stddev,
                                             wd=wd)
        # conv = conv2d(images, kernel, 2)
        conv = tf.nn.conv2d(bottom, kernel, strides=stride, padding=padding)
        biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(bConst))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.elu(bias, name=scope.name)
        _activation_summary(conv1)
        return conv1


def get_max_pool(name, bottom):
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def get_fc_layer(name, bottom, use_xavier, shape, stddev, wd, bConst, isflat, isdrop):
    keep_prob = 0.8
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             use_xavier,
                                             shape=shape,
                                             stddev=stddev,
                                             wd=wd)
        biases = _variable_on_cpu('biases', [shape[-1]], tf.constant_initializer(bConst))
        if isflat:
            conv_flat = tf.reshape(bottom, [-1, shape[0]])
            fc = tf.nn.elu(tf.matmul(conv_flat, kernel) + biases)
        else:
            fc = tf.nn.elu(tf.matmul(bottom, kernel) + biases)
        if isdrop:
            fc_drop = tf.nn.dropout(fc, keep_prob)
            _activation_summary(fc_drop)
            return fc_drop
        else:
            _activation_summary(fc)
            return fc

#This model works only when the input image size is 50x120x3
def inference_vgg(images):
  """Build the model similar to VGG.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  use_xavier = False
  keep_prob = 0.8
  stride = [1,1,1,1]
  conv1_1 = get_conv2d('conv1_1', images, use_xavier, [3, 3, 3, 64], 0.01, 0.00004, stride, 0.1,'SAME')
  conv1_2 = get_conv2d('conv1_2', conv1_1, use_xavier, [3, 3, 64, 64], 0.1, 0.00004, stride, 0.1, 'SAME')
  pool1 = get_max_pool('pool1', conv1_2)

  conv2_1 = get_conv2d('conv2_1', pool1, use_xavier, [3, 3, 64, 128], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv2_2 = get_conv2d('conv2_2', conv2_1, use_xavier, [3, 3, 128, 128], 0.01, 0.00004, stride, 0.1, 'SAME')
  pool2 = get_max_pool('pool2', conv2_2)

  conv3_1 = get_conv2d('conv3_1', pool2, use_xavier, [3, 3, 128, 256], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv3_2 = get_conv2d('conv3_2', conv3_1, use_xavier, [3, 3, 256, 256], 0.01, 0.00004, stride, 0.1, 'SAME')
  conv3_3 = get_conv2d('conv3_3', conv3_2, use_xavier, [3, 3, 256, 256], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv3_4 = get_conv2d('conv3_4', conv3_3, use_xavier, [3, 3, 256, 256], 0.01, 0.00004, stride, 0.1, 'SAME')
  pool3 = get_max_pool('pool3', conv3_4)

  conv4_1 = get_conv2d('conv4_1', pool3, use_xavier, [3, 3, 256, 512], 0.01, 0.00004, stride, 0.1, 'SAME')
  conv4_2 = get_conv2d('conv4_2', conv4_1, use_xavier, [3, 3, 512, 512], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv4_3 = get_conv2d('conv4_3', conv4_2, use_xavier, [3, 3, 512, 512], 0.01, 0.00004, stride, 0.1, 'SAME')
  conv4_4 = get_conv2d('conv4_4', conv4_3, use_xavier, [3, 3, 512, 512], 0.1, 0.00004, stride, 0.1, 'SAME')
  pool4 = get_max_pool('pool4', conv4_4)

  conv5_1 = get_conv2d('conv5_1', pool4, use_xavier, [3, 3, 512, 512], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv5_2 = get_conv2d('conv5_2', conv5_1, use_xavier, [3, 3, 512, 512], 0.01, 0.00004, stride, 0.1, 'SAME')
  conv5_3 = get_conv2d('conv5_3', conv5_2, use_xavier, [3, 3, 512, 512], 0.1, 0.00004, stride, 0.1, 'SAME')
  conv5_4 = get_conv2d('conv5_4', conv5_3, use_xavier, [3, 3, 512, 512], 0.01, 0.00004, stride, 0.1, 'VALID')
  pool5 = get_max_pool('pool5', conv5_4)

  fc1 = get_fc_layer('fc1', pool5, use_xavier, [2048, 512], 0.1, 0.0004, 0.1, True, False)
  fc2 = get_fc_layer('fc2', fc1, use_xavier, [512, 100], 0.01, 0.0004, 0.1, False, False)
  fc3 = get_fc_layer('fc3', fc2, use_xavier, [100, 10], 0.1, 0.0004, 0.1, False, False)

  # output
  with tf.variable_scope('output') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           use_xavier,
                                           shape=[10,1],
                                           stddev=0.01,
                                           wd=0.00004)
      biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
      #y = tf.matmul(fc3, kernel) + biases
      y = tf.mul(tf.atan(tf.matmul(fc3, kernel) + biases), 5)
      _activation_summary(y)

  return y

def loss(logits, steering_angle):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  loss_mse = tf.reduce_mean(tf.abs(tf.sub(logits, steering_angle)))
  tf.add_to_collection('losses', loss_mse)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in model.
  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train model.
  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.
  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size_flag
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdagradOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
