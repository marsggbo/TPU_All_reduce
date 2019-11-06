# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""MNIST model training using TPUs.

This program demonstrates training of the convolutional neural network model
defined in mnist.py on Google Cloud TPUs (https://cloud.google.com/tpu/).

If you are not interested in TPUs, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

# For open source environment, add grandparent directory for import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(sys.path[0]))))

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("use_bfloa16", False, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 5000,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")
tf.flags.DEFINE_integer("scale", 1, "Scale of model size.")

FLAGS = tf.flags.FLAGS


def metric_fn(labels, logits):
  accuracy = tf.metrics.accuracy(
      labels=labels, predictions=tf.argmax(logits, axis=1))
  return {"accuracy": accuracy}

def create_model(scale=1):
  '''
  scale=-1: model size=256KB
  scale=1: model size=512KB
  scale=2: model size=1MB
  '''
  l = tf.keras.layers
  input_shape = (512*scale) if scale>0 else (512//abs(scale))
  model = tf.keras.Sequential(
    [
      l.InputLayer(input_shape=input_shape),
      l.Dense(512, use_bias=False),
    ])
  model_size = model.count_params()*4/1024**2 if scale>0 else model.count_params()*4/1024
  model_size = str(model_size)+" MB" if scale>0 else str(model_size)+" KB"
  print("Scale: {} Model size: {}".format(scale, model_size))
  return model

def model_fn(features, labels, mode, params):
  """model_fn constructs the ML model used to predict handwritten digits."""

  scale = params["scale"]
  bs = params["bs"]
  del params
  image = tf.random.normal([bs, 512*scale]) if scale>0 else tf.random.normal([bs, 512//abs(scale)]) 
  labels = tf.constant(np.random.randint(0,512,[bs,1]))
  labels = tf.cast(labels, tf.int64)

  model = create_model(scale=FLAGS.scale)

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'class_ids': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)

  logits = model(image, training=(mode == tf.estimator.ModeKeys.TRAIN))
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_global_step(),
        decay_steps=100000,
        decay_rate=0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_global_step()))

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))


def main(argv):
  del argv  # Unused.
  tf.logging.set_verbosity(tf.logging.INFO)

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      params={"data_dir": FLAGS.data_dir, 'scale': FLAGS.scale, 'bs':FLAGS.batch_size},
      config=run_config)
  # TPUEstimator.train *requires* a max_steps argument.
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

def train_input_fn(params):
  """train_input_fn defines the input pipeline used for training."""
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Retrieves the batch size for the current shard. The # of shards is
  # computed according to the input pipeline deployment. See
  # `tf.contrib.tpu.RunConfig` for details.
  images = tf.random.normal([256, 512])
  labels = tf.random.normal([256, 1])
  labels = tf.cast(labels, tf.int64)
  return images, labels

if __name__ == "__main__":
  tf.app.run()
