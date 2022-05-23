import os
import re
import sys

import numpy as np
import tensorflow as tf
import dataset
import utils
import params_setting
import argparse

def train_val(params):
  params.batch_size = 1
  #utils.backup_python_files_and_params(params)

  train_datasets = []
  train_ds_iters = []
  max_train_size = 0
  for i in range(len(params.datasets2use['train'])):
    this_train_dataset, n_trn_items = dataset.tf_point_cloud_dataset(params, params.datasets2use['train'][i],
                                                              size_limit=params.train_dataset_size_limit,
                                                              shuffle_size=100,
                                                              n_models=np.inf,
                                                              data_augmentation=params.train_data_augmentation)
    print('Train Dataset size:', n_trn_items)
    train_ds_iters.append(iter(this_train_dataset.repeat()))
    train_datasets.append(this_train_dataset)
    max_train_size = max(max_train_size, n_trn_items)
  train_epoch_size = max(8, int(max_train_size / params.n_walks_per_model / params.batch_size))
  print('train_epoch_size:', train_epoch_size)

  test_dataset, n_tst_items = dataset.tf_point_cloud_dataset(params, params.datasets2use['test'][0],
                                                      size_limit=np.inf,
                                                      shuffle_size=100,
                                                      n_models=np.inf)
  test_ds_iter = iter(test_dataset.repeat())
  print(' Test Dataset size:', n_tst_items)

  def save_traj_as_npz(name, model_ftrs, labels):
    m = {}
    m['model_features'] = np.asarray(model_ftrs)
    m['label'] = labels
    m['model_id'] = str(name)

    sub_dir = re.split(string=str(name), pattern='\/')
    sub_dir = sub_dir[-1]
    sub_dir = re.split(string=sub_dir, pattern='.npz')
    sub_dir = sub_dir[0]

    save_path = params.ds_path_dirs + '/' + sub_dir + '/'
    if os.path.isdir(params.ds_path_dirs+'/') == False:
      os.mkdir(params.ds_path_dirs+'/')
    if os.path.isdir(save_path) == False:
      os.mkdir(save_path)
    file_name = os.path.join(save_path, sub_dir+'_traj_' + str(epoch)+'.npz')
    if os.path.isfile(file_name) == True:
      return
    np.savez(file_name, **m)

  params.batch_size = 1
  for epoch in range(0, 600, 1):
    for iter_db in range(train_epoch_size):
      for dataset_id in range(len(train_ds_iters)):
        name, model_ftrs, labels = train_ds_iters[dataset_id].next()
        save_traj_as_npz(name=name, model_ftrs=model_ftrs, labels=labels)

    if test_dataset is not None:
      for j in range(n_tst_items):
          name, model_ftrs, labels = test_ds_iter.next()
          save_traj_as_npz(name=name, model_ftrs=model_ftrs, labels=labels)

  return

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
  np.random.seed(1)
  utils.config_gpu()


  if len(sys.argv) <= 1:
    print('Use: python save_walk_as_npz.py <dataset name>')
  else:
    job = sys.argv[1]
    run_one_job(job)
