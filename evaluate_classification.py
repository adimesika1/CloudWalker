import os, sys, copy
from easydict import EasyDict
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from distinctipy import distinctipy
import time

import walks
import rnn_model
import utils
import dataset

def calc_accuracy_test(dataset_expansion=False, logdir=None, labels=None, iter2use='last', classes_indices_to_use=None,
                       dnn_model=None, params=None, n_models=np.inf, model_fn=None, n_walks_per_model=16, data_augmentation={}, seed=0):
  SHOW_WALK=0
  # Prepare parameters for the evaluation
  if params is None:
    with open(logdir + '/params.txt') as fp:
      params = EasyDict(json.load(fp))
  else:
    params = copy.deepcopy(params)
  if logdir is not None:
    params.logdir = logdir
  if model_fn is not None:
    pass
  elif iter2use != 'last':
    model_fn = logdir + '/learned_model2keep__' + iter2use + '.keras'
    model_fn = model_fn.replace('//', '/')
  else:
    model_fn = tf.train.latest_checkpoint(params.logdir)


  params.batch_size = 1
  params.n_walks_per_model = n_walks_per_model

  if SHOW_WALK:
    params.net_input += ['vertex_indices']
  test_dataset, n_tst_items = dataset.tf_pre_created_dataset(params, params.datasets2use['test'][0],
                                                             size_limit=params.test_dataset_size_limit,
                                                             permute_file_names=False,
                                                             shuffle_size=0)
  # If dnn_model is not provided, load it
  if dnn_model is None:
    if params.net == 'RnnWalkNet':
      dnn_model = rnn_model.RnnWalkNet(params, params.n_classes, params.net_input_dim-SHOW_WALK, model_fn, model_must_be_load=True,
                                       dump_model_visualization=False)

  n_classes = params.n_classes
  all_confusion = np.zeros((n_classes, n_classes), dtype=np.int)
  pred_per_model_name = {}

  # for retrieval
  all_features = []
  all_labels = []
  models_ids = []
  to_print=0
  for i, data in tqdm(enumerate(test_dataset), total=n_tst_items):
    tb = time.perf_counter()
    name, ftrs, gt = data
    model_fn = name.numpy()[0].decode()
    model_name = utils.get_model_name_from_npz_fn(model_fn)
    sp = ftrs.shape
    ftrs = tf.reshape(ftrs, (-1, sp[-2], sp[-1]))
    gt = gt.numpy()[0]
    ftr2use = ftrs.numpy()

    features, predictions = dnn_model(ftr2use, classify=False, training=False)
    pred_value, pred_count = np.unique(np.argmax(predictions, axis=1), return_counts=True)
    ind = np.argmax(pred_count)
    max_hit = pred_value[ind]
    pred_per_model_name[model_name] = [int(gt), max_hit]
    # generate confusion matrix
    all_confusion[int(gt), max_hit] += 1
    to_print += time.perf_counter()-tb
    # for retrieval
    #all_features.append(np.mean(predictions, axis=0) / np.linalg.norm(np.mean(predictions, axis=0)))
    #all_labels.append(gt)
    #models_ids.append(model_name)

  n_models = 0
  n_sucesses = 0
  prediction_per_model = []
  for k, v in pred_per_model_name.items():
    gt = v[0]
    pred = v[1]
    n_models += 1
    n_sucesses += pred == gt
    prediction_per_model.append(pred)
  overall_accuracy = n_sucesses / n_models
  print("average seconds per object: ", to_print/n_models)
  # calculate accuracy per category
  mean_acc_per_class = calculate_mean_accuracy_per_class(labels, all_confusion)

  #save_features_for_retrieval(all_features, all_labels, models_ids, params.logdir)

  return [overall_accuracy, mean_acc_per_class], dnn_model



def save_features_for_retrieval(features, labels, ids, logdir):
  np.savez(os.path.join(logdir, 'features.npz'),
           features=np.stack(features, axis=0),
           labels=np.asarray(labels),
           model_ids=ids)

def calculate_mean_accuracy_per_class(labels, all_confusion):
  acc_per_class = []
  for i, name in enumerate(labels):
    this_type = all_confusion[i]
    n_this_type = this_type.sum()
    accuracy_this_type = this_type[i] / n_this_type
    if n_this_type:
      acc_per_class.append(accuracy_this_type)
    this_type_ = this_type.copy()
    this_type_[i] = -1
  return np.mean(acc_per_class)

# For debugging
def print_confusion_matrix(all_confusion, labels, logdir):
  df_cm = pd.DataFrame(all_confusion, [c for c in labels],
                       [c for c in labels])
  sn.set_theme(style="white")
  f, ax = plt.subplots(figsize=(25, 20))
  cmap = sn.diverging_palette(230, 20, as_cmap=True)
  sn.heatmap(df_cm, cmap="BuPu", vmin=0, vmax=np.max(all_confusion), center=0,
             linewidths=.5, annot=True)
  plt.xlabel("predicted labels")
  plt.ylabel("gt labels")
  # plt.show()
  num_of_classes, _ = all_confusion.shape
  plt.savefig(logdir + '/' + str('confusion_modelnet') + str(num_of_classes) +'.png', dpi=400)

# For debugging
def plot_tsne(embeddings, prediction, num_categories, keys=None, output_path=None):
  # adapted from:https://towardsdatascience.com/visualizing-feature-vectors-embeddings-using-pca-and-t-sne-ef157cea3a42
  # Create a two dimensional t-SNE projection of the embeddings
  tsne = TSNE(2, verbose=1)
  tsne_proj = tsne.fit_transform(embeddings)

  cmap = distinctipy.get_colors(num_categories)
  cmap = np.concatenate([np.asarray(cmap), np.expand_dims(np.ones(len(cmap)), -1)], axis=-1)
  fig, ax = plt.subplots(figsize=(15, 15))
  for label in range(num_categories):
    indices = np.where(np.asarray(prediction) == label)
    lab_name = label if keys is None else keys[label]
    ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap[label]).reshape(1, 4), label=lab_name,
               alpha=0.5)
  ax.legend(fontsize='large', markerscale=2)
  plt.axis('off')
  plt.tight_layout()
  if output_path is not None:
    plt.savefig(output_path, dpi=400)
  else:
    plt.show()

# For debugging
def show_walk(model, walks,labels, predictions, color_inds =None):#,print_all =True):
  SHOW_LOGITS = 1
  colors = ['pink', 'lightblue', 'lime', 'orange', 'black', 'blue', 'yellow', 'red', 'lightgreen']
  for wi in range(walks.shape[0]):
    if color_inds is None:
      clr = wi
    else:
      clr = color_inds[wi]
    walk = walks[wi, :, -1].astype(np.int)
    utils.visualize_model_walk(model['vertices'], [list(walk)], color=colors[clr%(len(colors))])
    if SHOW_LOGITS:
      # creating the data
      l = list(labels)
      v = list(predictions[wi,:])
      fig = plt.figure(figsize=(10, 5))
      # creating the bar plot
      plt.bar(l, v, color='maroon',width=0.4)
      plt.xlabel("category")
      plt.ylabel("probability")
      plt.show()


if __name__ == '__main__':
  from train_val import get_params
  utils.config_gpu(True)

  if len(sys.argv) != 3:
    print('Use: python evaluate_classification.py <dataset> <trained model directory>')
    print('For example: python evaluate_classification.py modelnet40_normal_resampled runs/0012-31.03.2022..15.14__modelnet40_normal_resampled')
  else:
    logdir = sys.argv[2]
    job = sys.argv[1]
    params = get_params(job)
    acc_overall = []
    acc_per_class = []

    for i in range(10):
      seed1 = np.random.randint(1, 10000)
      seed2 = np.random.randint(1, 10000)
      #random.seed(seed1)
      #np.random.seed(seed2)
      #tf.random.set_seed(seed1)
      #random.seed(np.random.randint(1, 10000))
      np.random.seed(np.random.randint(1, 10000))
      tf.random.set_seed(np.random.randint(1, 10000))
      accs, _ = calc_accuracy_test(logdir=logdir,
                                   dataset_expansion=params.full_accuracy_test['dataset_expansion'],
                                   labels=params.full_accuracy_test['labels'],
                                   n_walks_per_model=params.full_accuracy_test['n_walks_per_model'],
                                   iter2use='00120037', params=params, seed = seed2)
      print('Mean accuracy:', accs[0])
      print('Mean per class accuracy:', accs[1])
      print('seed1: ',seed1)
      print('seed2: ',seed2)
      acc_overall.append(accs[0])
      acc_per_class.append(accs[1])
    print("list_overall: ", acc_overall)
    print("list_mean_per_class: ", acc_per_class)
    print("acc_overall: ", np.mean(acc_overall))
    print("acc_per_class: ", np.mean(acc_per_class))
    print("acc_overall_var: ", np.std(acc_overall))
    print("acc_per_class_var: ", np.std(acc_per_class))
    print("max_overall_acc: ", np.max(acc_overall))


    ### modelnet40_normal_resampled runs/0075-24.01.2022..11.05__modelnet40_normal_resampled ### iter2use=00120037
    ### 3dfuture runs/0314-13.03.2022..09.54__future3d ### iter2use=iter2use
