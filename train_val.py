import os

import evaluate_classification

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import rnn_model
import dataset
import utils
import params_setting


def train_val(params):
  utils.next_iter_to_keep = 10000
  print(utils.color.BOLD + utils.color.RED + 'params.logdir :::: ', params.logdir, utils.color.END)
  print(utils.color.BOLD + utils.color.RED, os.getpid(), utils.color.END)
  utils.backup_python_files_and_params(params)

  # Set up datasets for training and for test
  # -----------------------------------------
  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  for i in range(len(params.datasets2use['train'])):
    this_train_dataset, n_trn_items = dataset.tf_pre_created_dataset(params, params.datasets2use['train'][i],
                                                                     size_limit=params.train_dataset_size_limit,
                                                                     shuffle_size=100)
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)
    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)
  if params.datasets2use['test'] is None:
    test_dataset = None
    n_tst_items = 0
  else:
    test_dataset, n_tst_items = dataset.tf_pre_created_dataset(params, params.datasets2use['test'][0],
                                                               size_limit=params.test_dataset_size_limit,
                                                               shuffle_size=100)
    test_ds_iter = iter(test_dataset.repeat())
  print('Test Dataset size:', n_tst_items)

  # Set up RNN model and optimizer
  # ------------------------------
  if params.net_start_from_prev_net is not None:
    init_net_using = params.net_start_from_prev_net
  else:
    init_net_using = None

  if params.optimizer_type == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate[0], clipnorm=params.gradient_clip_th)
  elif params.optimizer_type == 'cycle':
    @tf.function
    def _scale_fn(x):
      x_th = 500e3 / params.cycle_opt_prms.step_size
      if x < x_th:
        return 1.0
      else:
        return 0.5
    lr_schedule = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params.cycle_opt_prms.initial_learning_rate,
                                                      maximal_learning_rate=params.cycle_opt_prms.maximal_learning_rate,
                                                      step_size=params.cycle_opt_prms.step_size,
                                                      scale_fn=_scale_fn, scale_mode="cycle", name="MyCyclicScheduler")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=params.gradient_clip_th)
  else:
    raise Exception('optimizer_type not supported: ' + params.optimizer_type)

  if params.net == 'RnnWalkNet':
    dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim, init_net_using, optimizer=optimizer)

  # Other initializations
  # ---------------------
  time_msrs = {}
  time_msrs_names = ['train_step', 'get_train_data', 'test']
  for name in time_msrs_names:
    time_msrs[name] = 0
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  train_log_names = ['train_loss']
  train_logs = {name: tf.keras.metrics.Mean(name=name) for name in train_log_names}
  train_logs['train_accuracy'] = train_accuracy

  # loss function
  # ----------------------
  train_loss = tf.keras.losses.SparseCategoricalCrossentropy()

  @tf.function
  def train_step(model_ftrs_, labels_):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    with tf.GradientTape() as tape:
      labels = tf.reshape(tf.transpose(tf.stack((labels_,)*params.n_walks_per_model)), (-1,))
      predictions = dnn_model(model_ftrs, classify=True, training=True)
      train_accuracy(labels, predictions)
      loss = train_loss(labels, predictions)
      loss += tf.reduce_sum(dnn_model.losses)

    gradients = tape.gradient(loss, dnn_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))

    train_logs['train_loss'](loss)

    return loss

  @tf.function
  def test_step(model_ftrs_, labels_):
    sp = model_ftrs_.shape
    model_ftrs = tf.reshape(model_ftrs_, (-1, sp[-2], sp[-1]))
    labels = tf.reshape(tf.transpose(tf.stack((labels_,) * params.n_walks_per_model)), (-1,))
    predictions = dnn_model(model_ftrs, classify=True, training=False)
    best_pred = tf.math.argmax(predictions, axis=-1)
    test_accuracy(labels, predictions)
    confusion = tf.math.confusion_matrix(labels=tf.reshape(labels, (-1,)), predictions=tf.reshape(best_pred, (-1,)),
                                         num_classes=params.n_classes)
    return confusion
  # -------------------------------------

  # Loop over training EPOCHs
  # -------------------------
  next_iter_to_log = 0
  e_time = 0
  accrcy_smoothed = tb_epoch = last_loss = None
  all_confusion = {}
  with tf.summary.create_file_writer(params.logdir).as_default():
    epoch = 0
    while optimizer.iterations.numpy() < params.iters_to_train + train_epoch_size * 2:
      epoch += 1
      str_to_print = str(os.getpid()) + ') Epoch' + str(epoch) + ', iter ' + str(optimizer.iterations.numpy())

      # Save some logs & infos
      utils.save_model_if_needed(optimizer.iterations, dnn_model, params)
      if tb_epoch is not None:
        e_time = time.time() - tb_epoch
        tf.summary.scalar('time/one_epoch', e_time, step=optimizer.iterations)
        tf.summary.scalar('time/av_one_trn_itr', e_time / n_iters, step=optimizer.iterations)
        for name in time_msrs_names:
          if time_msrs[name]:  # if there is something to save
            tf.summary.scalar('time/' + name, time_msrs[name], step=optimizer.iterations)
            time_msrs[name] = 0
      tb_epoch = time.time()
      n_iters = 0
      tf.summary.scalar(name="train/learning_rate", data=optimizer._decayed_lr(tf.float32), step=optimizer.iterations)
      tf.summary.scalar(name="mem/free", data=utils.check_mem_and_exit_if_full(), step=optimizer.iterations)
      gpu_tmpr = utils.get_gpu_temprature()
      tf.summary.scalar(name="mem/gpu_tmpr", data=gpu_tmpr, step=optimizer.iterations)

      # Train one EPOC
      str_to_print += '; LR: ' + str(optimizer._decayed_lr(tf.float32))
      train_logs['train_loss'].reset_states()
      tb = time.time()
      for iter_db in range(train_epoch_size):
        for dataset_id in range(len(train_datasets)):
          name, model_ftrs, labels = train_ds_iters[dataset_id].next()
          dataset_type = utils.get_dataset_type_from_name(name)
          time_msrs['get_train_data'] += time.time() - tb

          n_iters += 1
          tb = time.time()
          train_step(model_ftrs, labels)
          loss2show = 'train_loss'

          time_msrs['train_step'] += time.time() - tb
          tb = time.time()
        if iter_db == train_epoch_size - 1:
          str_to_print += ', TrnLoss: ' + str(round(train_logs[loss2show].result().numpy(), 2))

      # Dump training info to tensorboard
      if optimizer.iterations >= next_iter_to_log:
        for k, v in train_logs.items():
          if v.count.numpy() > 0:
            tf.summary.scalar('train/' + k, v.result(), step=optimizer.iterations)
            v.reset_states()
        next_iter_to_log += params.log_freq

      # Run test on part of the test set
      if test_dataset is not None:
        n_test_iters = 0
        tb = time.time()
        for i in range(n_tst_items):
          name, model_ftrs, labels = test_ds_iter.next()
          n_test_iters += model_ftrs.shape[0]
          if n_test_iters > params.n_models_per_test_epoch:
            break
          confusion = test_step(model_ftrs, labels)
          dataset_type = utils.get_dataset_type_from_name(name)
          if dataset_type in all_confusion.keys():
            all_confusion[dataset_type] += confusion
          else:
            all_confusion[dataset_type] = confusion
        # Dump test info to tensorboard
        if accrcy_smoothed is None:
          accrcy_smoothed = test_accuracy.result()
        accrcy_smoothed = accrcy_smoothed * .9 + test_accuracy.result() * 0.1
        tf.summary.scalar('test/accuracy_' + dataset_type, test_accuracy.result(), step=optimizer.iterations)
        tf.summary.scalar('test/accuracy_smoothed', accrcy_smoothed, step=optimizer.iterations)
        str_to_print += ', test/partial_test_accuracy: ' + str(round(test_accuracy.result().numpy(), 2))
        time_msrs['test'] += time.time() - tb
        # best local acc
        '''if test_accuracy.result().numpy() > params.best_local_acc:
          if os.path.isdir(params.logdir + '/local_best/') == False:
            os.mkdir(params.logdir + '/local_best/')
          params.best_local_acc = test_accuracy.result().numpy()
          if params.best_local_acc >= 0.92:
            dnn_model.save_weights(params.logdir + '/local_best/', optimizer.iterations.numpy(), keep=True)
            local_accuracy, _ = evaluate_classification.calc_accuracy_test(params=params, dnn_model=dnn_model,
                                                                       **params.full_accuracy_test)
            print(local_accuracy)'''
        test_accuracy.reset_states()
      str_to_print += ', time: ' + str(round(e_time, 1))
      print(str_to_print)


  return last_loss


def get_params(job):
  # Classifications
  job = job.lower()
  if job == 'modelnet40_normal_resampled':
    params = params_setting.modelnet40_normal_resampled_params()

  if job == 'modelnet10_normal_resampled':
    params = params_setting.modelnet10_normal_resampled_params()

  if job == '3dfuture':
    params = params_setting.future3d_params()

  if job == 'scanobjectnn':
    params = params_setting.scanobjectnn_params()

  return params

def run_one_job(job):
  params = get_params(job)
  train_val(params)


if __name__ == '__main__':
  np.random.seed(0)
  utils.config_gpu()


  if len(sys.argv) <= 1:
    print('Use: python train_val.py <job>')
  else:
    job = sys.argv[1]

    run_one_job(job)

