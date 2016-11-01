from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import predict_steering

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/sridhar/code/Challenge/Code_tfRecord/model_deg5/eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'train_eval',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/sridhar/code/Challenge/Code_tfRecord/model_deg5/',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 3812,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


def eval_once(saver, summary_writer, mse_op, logits_op, steering_angles_op, summary_op):
  """Run Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    mse: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size_flag))
      s_mse = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size_flag
      step = 0
      while step < num_iter and not coord.should_stop():
        mse_predict, steering_predict, steering_angles = sess.run([mse_op, logits_op, steering_angles_op])
        s_mse += np.sum(mse_predict)
        step += 1

      # Compute precision @ 1.
      average_mse = s_mse / total_sample_count
      print('%s: Average MSE = %.3f' % (datetime.now(), average_mse))
      print('%s: Sum MSE = %.3f' % (datetime.now(), s_mse))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Average MSE', simple_value=average_mse)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels
    eval_data = FLAGS.eval_data == 'test'
    images, steering_angles = predict_steering.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = predict_steering.inference_nvidia(images)

    # Calculate predictions.
    mse = tf.reduce_mean(tf.square(tf.sub(logits, steering_angles)),name='mse')
    #top_k_op = tf.nn.in_top_k(logits, steering_angles, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        predict_steering.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    #tf.scalar_summary('steering_predict',logits)
    tf.scalar_summary(mse.op.name,mse)

    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, mse, logits, steering_angles, summary_op)
      if FLAGS.run_once:
        break
      #time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()