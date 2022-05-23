from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from easydict import EasyDict
import copy

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob

import utils

from tensorflow import keras
layers = tf.keras.layers

class RnnWalkBase(tf.keras.Model):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn=None,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnWalkBase, self).__init__(name='')

    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load
    self._init_layers()
    inputs = tf.keras.layers.Input(shape=(100, net_input_dim))
    self.build(input_shape=(1, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=inputs, outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/RnnWalkModel.png', show_shapes=True)
    self.manager = None
    if optimizer:
      if model_fn:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      else:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
      self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=5)
      if model_fn: # Transfer learning
        self.load_weights(model_fn)
        self.checkpoint.optimizer = optimizer
      else:
        self.load_weights()
    else:
      self.checkpoint = tf.train.Checkpoint(model=self)
      if model_fn is None:
        model_fn = self._get_latest_keras_model()
      self.load_weights(model_fn)

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def _get_latest_keras_model(self):
    filenames = glob.glob(self._params.logdir + '/*model2keep__*')
    iters_saved = [int(f.split('model2keep__')[-1].split('.keras')[0]) for f in filenames]
    return filenames[np.argmax(iters_saved)]


  def load_weights(self, filepath=None):
    if filepath is not None and filepath.endswith('.keras'):
      super(RnnWalkBase, self).load_weights(filepath)
    elif filepath is None:
      _ = self.checkpoint.restore(self.manager.latest_checkpoint)
      print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(), utils.color.END)
    else:
      filepath = filepath.replace('//', '/')
      _ = self.checkpoint.restore(filepath)


  def save_weights(self, folder, step=None, keep=False):
    if self.manager is not None:
      self.manager.save()
    if keep:
      super(RnnWalkBase, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')


class RnnWalkNet(RnnWalkBase):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 64, 'fc2': 128, 'fc3': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512, 'fc4': 512, 'fc5': 128}
    else:
      self._layer_sizes = params.layer_sizes
    super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization, optimizer=optimizer)

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
      self._norm1 = tfa.layers.InstanceNormalization(axis=2)
      self._norm2 = tfa.layers.InstanceNormalization(axis=2)
      self._norm3 = tfa.layers.InstanceNormalization(axis=2)
      self._norm4 = tfa.layers.InstanceNormalization(axis=2)
    elif self._params.use_norm_layer == 'BatchNorm':
      self._norm1 = layers.BatchNormalization(axis=2)
      self._norm2 = layers.BatchNormalization(axis=2)
      self._norm3 = layers.BatchNormalization(axis=2)
      self._norm4 = layers.BatchNormalization(axis=2)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc3 = layers.Dense(self._layer_sizes['fc3'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc4 = layers.Dense(self._layer_sizes['fc4'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)

    rnn_layer = layers.GRU
    self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                            dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                            dropout=self._params.net_gru_dropout,
                            recurrent_initializer=initializer, kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer)
    self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                           return_sequences=not self._params.one_label_per_model,
                           return_state=False,
                           dropout=self._params.net_gru_dropout,
                           recurrent_initializer=initializer, kernel_initializer=initializer,
                           kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                           bias_regularizer=kernel_regularizer)
    self._fc_last = layers.Dense(self._classes, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer, activation='softmax')

  def call(self, model_ftrs, classify=True, training=True):

    x = model_ftrs[:, :]
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc3(x)
    if self._use_norm_layer:
      x = self._norm3(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc4(x)
    if self._use_norm_layer:
      x = self._norm4(x, training=training)
    x = tf.nn.relu(x)


    x = tf.concat([model_ftrs[:, :, 0:3], x], axis=-1)
    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)

    f = x3

    x = self._fc_last(f, training=training)
    if classify:
      return x
    else:
      return f,x


def show_model():
  def fn(to_print):
    print(to_print)
  if 1:
    params = EasyDict({'n_classes': 3, 'net_input_dim': 3, 'batch_size': 32, 'last_layer_activation': 'softmax',
                       'one_label_per_model': True, 'logdir': '.', 'layer_sizes': None, 'use_norm_layer': 'InstanceNorm',
                       'net_gru_dropout':0})
    params.net_input_dim = 3 + 5
    model = RnnWalkNet(params, classes=3, net_input_dim=3, model_fn=None)
  else:
    model = set_up_rnn_walk_model()
    tf.keras.utils.plot_model(model, "RnnWalkModel.png", show_shapes=True)
    model.summary(print_fn=fn)


def show_model():
  def fn(to_print):
    print(to_print)
  if 1:
    params = EasyDict({'n_classes': 3, 'net_input_dim': 3, 'batch_size': 32, 'last_layer_activation': 'softmax',
                       'one_label_per_model': True, 'logdir': '.', 'layer_sizes': None, 'use_norm_layer': 'InstanceNorm',
                       'net_gru_dropout':0})
    params.net_input_dim = 3 + 5
    model = RnnWalkWeightsNet(params, classes=3, net_input_dim=3, model_fn=None)
  else:
    model = set_up_rnn_walk_model()
    tf.keras.utils.plot_model(model, "RnnWalkWeightsNet.png", show_shapes=True)
    model.summary(print_fn=fn)

if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(0)
  show_model()
