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
        #self.checkpoint = tf.train.Checkpoint(optimizer=copy.deepcopy(optimizer), model=self)
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
      #if model_fn is None:
      #  model_fn = self._get_latest_keras_model()
      #self.load_weights(model_fn)
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
      #self.checkpoint.write(folder + '/learned_model2keep--' + str(step))

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
    self._norm5 = layers.BatchNormalization(axis=1)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc3 = layers.Dense(self._layer_sizes['fc3'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc4 = layers.Dense(self._layer_sizes['fc4'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc5 = layers.Dense(self._layer_sizes['fc5'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
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
    self._conv1 = layers.Conv1D(self._layer_sizes['fc2'], kernel_size=1, input_shape=[self._params.seq_len, self._params.net_input_dim])
    self._conv2 = layers.Conv1D(self._layer_sizes['fc3'], kernel_size=1, input_shape=[self._params.seq_len, self._layer_sizes['fc2']])
    self._conv3 = layers.Conv1D(self._layer_sizes['fc4'], kernel_size=1, input_shape=[self._params.seq_len, self._layer_sizes['fc3']])

  def call(self, model_ftrs, classify=True, training=True):

    x = model_ftrs[:, :] # 3 for 3dfuture
    #if self._params.use_model_scale:
    #s = model_ftrs[:, :, 6:7] #-2:-1


    '''x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)'''
    #x = self._conv1(x)
    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)

    #x = self._conv2(x)
    x = self._fc3(x)
    if self._use_norm_layer:
      x = self._norm3(x, training=training)
    x = tf.nn.relu(x)

    #x = self._conv3(x)
    x = self._fc4(x)
    if self._use_norm_layer:
      x = self._norm4(x, training=training)
    x = tf.nn.relu(x)


    #x = tf.concat([x ,model_ftrs[:, :, 0:3]], axis=-1) # switch x side for 3dfuture
    x = tf.concat([model_ftrs[:, :, 0:3], x], axis=-1)
    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)
    #if self._params.use_model_scale:
    #    x3 = tf.concat([s, x3], axis=-1)

    f = x3

    x = self._fc_last(f, training=training)
    if classify:
      return x
    else:
      return f,x

  def call_neighbors(self, model_ftrs, classify=True, skip_1st=True, training=True):
    neigs_inx = 6+self._params.num_of_neighbors_kdtree*3
    if skip_1st:
        x_n = model_ftrs[:, 1:, 0:6]
        neigs = model_ftrs[:, 1:, 6:neigs_inx]
        neigs_n = model_ftrs[:, 1:, neigs_inx:]
    else:
        x_n = model_ftrs[:, :, 0:3]
        neigs = model_ftrs[:, :, 6:neigs_inx]
        neigs_n = model_ftrs[:, :, neigs_inx:]

    num_of_neigs = tf.TensorShape([int(neigs.shape[-1] / 3)])
    temp_dim = copy.deepcopy(x_n.shape.dims)
    temp_dim[-1] = tf.compat.v1.Dimension(x_n.shape.dims[-1] // 2)
    neigs_dim = tf.TensorShape(temp_dim + num_of_neigs.dims)
    if all(neigs_dim):
        reshaped_neig = tf.reshape(neigs, neigs_dim)
        reshaped_neig_n = tf.reshape(neigs_n, neigs_dim)
        concat_neig_n = tf.concat([reshaped_neig, reshaped_neig_n], axis=2)
        x_n_neighs = tf.concat(values=[tf.expand_dims(x_n[:, :, 0:6], -1), concat_neig_n], axis=-1) #tf.expand_dims(n, -1),
        x_n_neighs = self._permute(x_n_neighs)
        x = self._fc1(x_n_neighs)
    else:
        x = x_n
        x = self._fc1(x)
    if all(neigs_dim) and self._use_norm_layer:
        x = self._norm1(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc2(x)
    if all(neigs_dim) and self._use_norm_layer:
        x = self._norm2(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc3(x)
    if all(neigs_dim) and self._use_norm_layer:
        x = self._norm3(x, training=training)
    x = tf.nn.relu(x)

    if all(neigs_dim):
        x = self._average_pooling(x)
        x = tf.squeeze(x)
        if x.shape.ndims < 3:
            x = tf.expand_dims(x, axis=0)
    x = tf.concat([x, x_n], axis=2)

    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)
    x = x3

    if classify:
      x = self._fc_last(x, training=training)

    return x


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


class RnnWalkWeightsNet(RnnWalkBase):
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
    super(RnnWalkWeightsNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load=model_must_be_load,
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
    self._norm5 = layers.BatchNormalization(axis=1)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc3 = layers.Dense(self._layer_sizes['fc3'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc4 = layers.Dense(self._layer_sizes['fc4'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc5 = layers.Dense(self._layer_sizes['fc5'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
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
    self._fc_last = layers.Dense(1, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer, activation=self._params.last_layer_activation)


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
    model = RnnWalkWeightsNet(params, classes=3, net_input_dim=3, model_fn=None)
  else:
    model = set_up_rnn_walk_model()
    tf.keras.utils.plot_model(model, "RnnWalkWeightsNet.png", show_shapes=True)
    model.summary(print_fn=fn)

class RnnExpertWalkNet(tf.keras.layers.Layer):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnExpertWalkNet, self).__init__()
    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 64, 'fc2': 128, 'fc3': 256, 'gru1': 256, 'gru2': 256, 'gru3': 128, 'fc4': 128, 'fc5': 64}
    else:
      self._layer_sizes = params.layer_sizes

    self._init_layers()

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.00001)
    initializer = tf.initializers.random_normal() #tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
        self._norm1 = tfa.layers.InstanceNormalization(axis=2)
        self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        self._norm3 = tfa.layers.InstanceNormalization(axis=2)
        self._norm4 = tfa.layers.InstanceNormalization(axis=1)
        self._norm5 = tfa.layers.InstanceNormalization(axis=1)
    elif self._params.use_norm_layer == 'BatchNorm':
        self._norm1 = layers.BatchNormalization(axis=2)
        self._norm2 = layers.BatchNormalization(axis=2)
        self._norm3 = layers.BatchNormalization(axis=2)
        self._norm4 = layers.BatchNormalization(axis=1)
        self._norm5 = layers.BatchNormalization(axis=1)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc3 = layers.Dense(self._layer_sizes['fc3'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc4 = layers.Dense(self._layer_sizes['fc4'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc5 = layers.Dense(self._layer_sizes['fc5'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
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
    self._scale_fc = layers.Dense(self._layer_sizes['gru3'], activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')


  def call(self, model_ftrs, classify=True, training=True):
    x = model_ftrs[:, :,0:6]
    only_x = model_ftrs[:, :,0:3]
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)

    x = tf.concat([x, only_x], axis=-1)
    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)
    f = x3

    x = self._fc_last(x3)
    return f,x


class RnnGateWalkNet(tf.keras.layers.Layer):
  def __init__(self,
               params,
               classes,
               net_input_dim,
               model_fn,
               model_must_be_load=False,
               dump_model_visualization=True,
               optimizer=None):
    super(RnnGateWalkNet, self).__init__()
    self._classes = classes
    self._params = params
    self._model_must_be_load = model_must_be_load
    if params.layer_sizes is None:
      self._layer_sizes = {'fc1': 64, 'fc2': 128, 'fc3': 256, 'gru1': 512, 'gru2': 512, 'gru3': 256, 'fc4': 128, 'fc5': 64}
    else:
      self._layer_sizes = params.layer_sizes
    self._init_layers()

  def _print_fn(self, st):
    with open(self._params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')

  def _init_layers(self):
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.random_normal() #tf.initializers.Orthogonal(3)
    self._use_norm_layer = self._params.use_norm_layer is not None
    if self._params.use_norm_layer == 'InstanceNorm':
        self._norm1 = tfa.layers.InstanceNormalization(axis=2)
        self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        self._norm3 = tfa.layers.InstanceNormalization(axis=2)
        #self._norm4 = tfa.layers.InstanceNormalization(axis=1)
        #self._norm5 = tfa.layers.InstanceNormalization(axis=1)
    elif self._params.use_norm_layer == 'BatchNorm':
        self._norm1 = layers.BatchNormalization(axis=2)
        self._norm2 = layers.BatchNormalization(axis=2)
        self._norm3 = layers.BatchNormalization(axis=2)
    self._norm4 = layers.BatchNormalization(axis=1)
    self._norm5 = layers.BatchNormalization(axis=1)
    self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc3 = layers.Dense(self._layer_sizes['fc3'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc4 = layers.Dense(self._layer_sizes['fc4'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc5 = layers.Dense(self._layer_sizes['fc5'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
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
    self._scale_fc = layers.Dense(self._layer_sizes['gru3'], activation=self._params.last_layer_activation, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._fc_last = layers.Dense(self._classes, activation='relu', kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')


  def call(self, model_ftrs, classify=True, training=True):
    x = model_ftrs[:, :,0:6]
    only_x = model_ftrs[:, :,0:3]
    x = self._fc1(x)
    if self._use_norm_layer:
      x = self._norm1(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc2(x)
    if self._use_norm_layer:
      x = self._norm2(x, training=training)
    x = tf.nn.relu(x)

    x = self._fc3(x)
    if self._use_norm_layer:
      x = self._norm3(x, training=training)
    x = tf.nn.relu(x)
    x = tf.concat([x, only_x], axis=-1)
    x1 = self._gru1(x, training=training)
    x2 = self._gru2(x1, training=training)
    x3 = self._gru3(x2, training=training)
    x = x3

    return x


class RnnCouncilWalker(tf.keras.Model):
  def __init__(self,
               params=None,
               classes=30,
               net_input_dim=3,
               optimizer_rnn=None,
               model_must_be_load=False,
               model_fn=None,
               dump_model_visualization=True,
               num_members=3,
               use_scale=False,
               ):
    super(RnnCouncilWalker, self).__init__()
    self.params = params
    self.num_members = num_members
    self.use_scale = use_scale
    self._rnn_1 = RnnExpertWalkNet(params=params,
                                     classes=classes,
                                     net_input_dim=net_input_dim,
                                     model_fn=model_fn,
                                     model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization,
                                     optimizer=optimizer_rnn)
    self._rnn_2 = RnnExpertWalkNet(params=params,
                                     classes=classes,
                                     net_input_dim=net_input_dim,
                                     model_fn=model_fn,
                                     model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization,
                                     optimizer=optimizer_rnn)
    self._rnn_3 = RnnExpertWalkNet(params=params,
                                     classes=classes,
                                     net_input_dim=net_input_dim,
                                     model_fn=model_fn,
                                     model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization,
                                     optimizer=optimizer_rnn)
    self._rnn_4 = RnnGateWalkNet(params=params,
                                     classes=num_members,
                                     net_input_dim=net_input_dim,
                                     model_fn=model_fn,
                                     model_must_be_load=model_must_be_load,
                                     dump_model_visualization=dump_model_visualization,
                                     optimizer=optimizer_rnn)

    self._argmax = tf.keras.backend.argmax
    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.random_normal() #tf.initializers.Orthogonal(3)
    self._norm4 = layers.BatchNormalization(axis=1)
    self._norm5 = layers.BatchNormalization(axis=1)
    self._fc4 = layers.Dense(128, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc5 = layers.Dense(64, kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                             kernel_initializer=initializer)
    self._fc_last = layers.Dense(num_members, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
    inputs = tf.keras.layers.Input(shape=(1, 100, net_input_dim))
    self.build(input_shape=(num_members, 1, 100, net_input_dim))
    outputs = self.call(inputs)
    if dump_model_visualization:
      tmp_model = keras.Model(inputs=[inputs], outputs=outputs, name='WalkModel')
      tmp_model.summary(print_fn=self._print_fn)
      tf.keras.utils.plot_model(tmp_model, params.logdir + '/Manager_RnnWalkModel.png', show_shapes=True)

    self.manager = None
    if optimizer_rnn:
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer_rnn, model=self)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.params.logdir, max_to_keep=1)
        if model_fn:  # Transfer learning
            self.load_weights(model_fn)
            self.checkpoint.optimizer = optimizer_rnn
        else:
            self.load_weights()
    else:
        self.checkpoint = tf.train.Checkpoint(model=self)
        if model_fn is None:
            model_fn = self._get_latest_keras_model()
        self.load_weights(model_fn)


  def _get_latest_keras_model(self):
    filenames = glob.glob(self._params.logdir + '/*model2keep__*')
    iters_saved = [int(f.split('model2keep__')[-1].split('.keras')[0]) for f in filenames]
    return filenames[np.argmax(iters_saved)]

  def load_weights(self, filepath=None):
    if filepath is not None and filepath.endswith('.keras'):
      super(RnnCouncilWalker, self).load_weights(filepath)
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
      super(RnnCouncilWalker, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')


  def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
    model_ftrs1 = model_ftrs[0]
    model_ftrs2 = model_ftrs[1]
    model_ftrs3 = model_ftrs[2]
    f1, rnn_1_pred = self._rnn_1.call(model_ftrs=model_ftrs1, training=training, classify=classify)
    f2, rnn_2_pred = self._rnn_2.call(model_ftrs=model_ftrs2, training=training, classify=classify)
    f3, rnn_3_pred = self._rnn_3.call(model_ftrs=model_ftrs3, training=training, classify=classify)
    rnn_4_pred_1 = self._rnn_4.call(model_ftrs=model_ftrs1, training=training)
    rnn_4_pred_2 = self._rnn_4.call(model_ftrs=model_ftrs2, training=training)
    rnn_4_pred_3 = self._rnn_4.call(model_ftrs=model_ftrs3, training=training)

    x = tf.keras.layers.Concatenate(axis=1)([rnn_4_pred_1, rnn_4_pred_2, rnn_4_pred_3])
    x = self._fc4(x)
    x = self._norm4(x, training=training)
    x = tf.nn.relu(x)
    x = self._fc5(x)
    x = self._norm5(x, training=training)
    x = tf.nn.relu(x)
    rnn_4_pred = self._fc_last(x)

    if training == True:
        return rnn_1_pred, rnn_2_pred, rnn_3_pred, rnn_4_pred
    else:
        gate_rnn_pred = self._argmax(rnn_4_pred, axis=1)
        len = (gate_rnn_pred).shape[0]
        if len is not None and len!=1:
            rnn1_rnn2_rnn3 = tf.stack([rnn_1_pred, rnn_2_pred, rnn_3_pred], axis=1)
            mask = np.zeros(shape=rnn1_rnn2_rnn3.shape, dtype=float)
            mask[:, gate_rnn_pred,:] = 1
            masked_pred = tf.multiply(rnn1_rnn2_rnn3, mask)
            #cp1 = tf.multiply(rnn_1_pred,tf.expand_dims(rnn_4_pred[:,0],axis=-1))#tf.math.reduce_max(masked_pred, axis=1)
            #cp2 = tf.multiply(rnn_2_pred,tf.expand_dims(rnn_4_pred[:,1],axis=-1))
            #cp3 = tf.multiply(rnn_3_pred,tf.expand_dims(rnn_4_pred[:,2],axis=-1))
            #combined_prediction = tf.multiply(cp1, cp2)
            #combined_prediction = tf.multiply(combined_prediction, cp3)
            combined_prediction = tf.math.reduce_max(masked_pred, axis=1)
            if classify:
                return combined_prediction
            f1_f2_f3 = tf.stack([f1, f2, f3], axis=1)
            mask = np.zeros(shape=f1_f2_f3.shape, dtype=float)
            mask[:, gate_rnn_pred,:] = 1
            masked_pred = tf.multiply(f1_f2_f3, mask)
            combined_features = tf.math.reduce_max(masked_pred, axis=1)
            return combined_features, combined_prediction, rnn_4_pred
        else:
            rnn_3_pred

  def _print_fn(self, st):
    with open(self.params.logdir + '/log.txt', 'at') as f:
      f.write(st + '\n')


if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu(0)
  show_model()
