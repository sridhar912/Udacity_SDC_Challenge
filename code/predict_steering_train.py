from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import predict_steering
from predict_steering import inference_nvidia

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/sridhar/code/Challenge/Code_tfRecord/model_deg0/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir','/home/sridhar/code/Challenge/Code_tfRecord/model_deg0/',
                           """Directory where to read model checkpoints.""")


def train():
  """Train for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels
    images, steering_labels, steering_angles = predict_steering.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = predict_steering.inference(images)
    #logits, angles = predict_steering.inference_nvidia(images,is_training=True)
    logits, angles = predict_steering.inference_nvidia(images, is_training=True)

    # Calculate loss.
    loss = predict_steering.loss(logits, steering_labels, angles, steering_angles)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = predict_steering.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Restored from checkpoint')
    else:
      print('No checkpoint file found. Started training from stratch')

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      #steering_labels1, steering_angles1,  logits_value, angles_value = sess.run([steering_labels, steering_angles, logits, angles])
      #mse_value, ce_value = sess.run([mse, ce])
      #s = np.subtract(angles_value, steering_angles1)
      #a = np.square(s)
      #m = np.mean(a)
      #loss_mse = sess.run([tf.reduce_mean(tf.square(tf.sub(angles, steering_angles)))])

      #label_v, logits_v, angles_v, _ = sess.run([steering_labels, logits, angles, train_op])

      start_time = time.time()
      _, loss_value= sess.run([train_op, loss])
      duration = time.time() - start_time

      #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size_flag
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()