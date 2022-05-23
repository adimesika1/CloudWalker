import numpy as np
import os

import sklearn
from sklearn.metrics.pairwise import euclidean_distances
import heapq
from sklearn.metrics import average_precision_score
# import ipdb

def plot_pr_cure(mpres, mrecs):
    pr_curve = np.zeros(mpres.shape[0], 10)
    for r in range(mpres.shape[0]):
        this_mprec = mpres[r]
        for c in range(10):
            pr_curve[r, c] = np.max(this_mprec[mrecs[r]>(c-1)*0.1])
    return pr_curve

def Eu_dis_mat_fast(X):
  aa = np.sum(np.multiply(X, X), 1)
  ab = X * X.T
  D = aa + aa.T - 2 * ab
  D[D < 0] = 0
  D = np.sqrt(D)
  D = np.maximum(D, D.T)
  return D

def calc_precision_recall(r):
  '''

  :param r:  ranked array of retrieved objects - '1' if we retrieved the correct label, '0' otherwise
  :return:
  '''
  num_gt = np.sum(r)   # total number of GT in class
  trec_precision = np.array([np.mean(r[:i+1]) for i in range(r.shape[0]) if r[i]])
  recall = [np.sum(r[:i+1]) / num_gt for i in range(r.shape[0])]
  precision = [np.mean(r[:i + 1]) for i in range(r.shape[0])]

  # interpolate it
  mrec = np.array([0.] + recall + [1.])
  mpre = np.array([0.] + precision + [0.])

  for i in range(len(mpre) - 2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i + 1])

  i = np.where(mrec[1:] != mrec[:-1])[0] + 1  # Is this a must? why not sum all?
  ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])   # area under the PR graph, according to information retrieval
  return trec_precision, mrec, mpre, ap

def calculate_map_auc(fts, lbls, model_ids, dis_mat=None, logdir=None):
  if dis_mat is None:
    dis_mat = sklearn.metrics.pairwise.euclidean_distances(np.mat(fts))
    #dis_mat = sklearn.metrics.pairwise.cosine_distances(np.mat(fts))
  num = len(lbls)
  mAP = 0
  trec_precisions = []
  mrecs = []
  mpres = []
  aps = []

  for i in range(num):
    scores = dis_mat[:, i]
    targets = (lbls == lbls[i]).astype(np.uint8)
    sortind = np.argsort(scores, 0)   #[:top_k]
    truth = targets[sortind]
    res = calc_precision_recall(truth)
    trec_precisions.append(res[0])
    mrecs.append(res[1])
    mpres.append(res[2])
    aps.append(res[3])

    #print to file
    ret_mid_list = np.asarray(model_ids)[sortind]
    if logdir is not None and os.path.exists(logdir):
      fold = logdir + '/retrieval_by_classification' +'/eu_dis/'
      file = fold + str(model_ids[i]).split('\\')[1]
      if not os.path.exists(fold):
        os.makedirs(fold)
      with(open(file, 'w+')) as f:
        for val in ret_mid_list[:1000]:
          f.write(str(val) + '\n')

  trec_precisions = np.concatenate(trec_precisions)
  mrecs = np.stack(mrecs)
  mpres = np.stack(mpres)
  # weighted sum of PR AUC per class
  aps = np.stack(aps)
  AUC = np.mean(aps)   # # according to 3D-ShapeNet   (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298801)
  mAP = np.mean(trec_precisions)  # according to 3D-ShapeNet   (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298801)
  return {'AUC': AUC, 'MAP': mAP}

def calculate_map_auc_by_classification(fts, lbls, models_ids=None, dis_mat=None, top_k=None, logdir=None):
  num = len(models_ids)
  mAP = 0
  trec_precisions = []
  mrecs = []
  mpres = []
  aps = []
  tested_models_ids = []
  sorted_inds = []
  # get mean prediction for each object individually
  all_preds = []
  all_lbls = []
  all_scores = []
  all_mids = []
  visited = []
  for i in range(num):
    if i in visited:
      continue
    scores = fts[i, :]
    # if we have multiple objects query per model - need to accumulate distances (scores) for correct prediction
    if models_ids is not None:
      cur_model_id = models_ids[i]
      all_model_i = np.where(models_ids == cur_model_id)
      assert np.min(lbls[all_model_i]) == np.max(lbls[all_model_i])
      visited += list(all_model_i[0])
      # TODO: skip next time we encountr that index
      if len(all_model_i) > 1:
        scores = np.mean(fts[all_model_i, :], axis=0)
    cur_pred = np.argmax(scores)  # Get classification of current object
    all_preds.append(cur_pred)
    all_scores.append(scores)
    all_lbls.append(lbls[i])
    all_mids.append(models_ids[i])

  all_scores=np.asarray(all_scores)
  all_lbls = np.asarray(all_lbls)
  for i in range(len(all_lbls)):
    # Retrieve list according to prediction
    # TODO: to add the rest of the data according to probability of that class
    sortind = np.argsort(all_scores[:, all_preds[i]])
    if top_k is not None:
      sortind = sortind[:top_k]
    targets = (all_lbls == all_lbls[i]).astype(np.uint8)
    # TODO: save mids for each model for estimating retrieval using official code

    truth = targets[sortind][::-1]
    # assert len(truth) == 2468
    ret_mid_list = np.asarray(all_mids)[sortind][::-1]
    if logdir is not None and os.path.exists(logdir):
      fold = logdir + '/retrieval_by_classification' +'/test_normal/'
      file = fold + str(all_mids[i]).split('\\')[1]
      if not os.path.exists(fold):
        os.makedirs(fold)
      with(open(file, 'w+')) as f:
        for val in ret_mid_list[:1000]:
          f.write(str(val) + '\n')

    res = calc_precision_recall(truth)
    trec_precisions.append(res[0])
    mrecs.append(res[1])
    mpres.append(res[2])
    aps.append(res[3])

  trec_precisions = np.concatenate(trec_precisions)
  # mrecs = np.stack(mrecs)
  # mpres = np.stack(mpres)
  # weighted sum of PR AUC per class
  aps = np.stack(aps)
  AUC = np.mean(aps)
  mAP = np.mean(trec_precisions)
  return {'AUC': AUC, 'mAP': mAP}, sorted_inds

def calculate_Top2(fts, lbls, models_ids=None, dis_mat=None, top_k=None, logdir=None):
  if dis_mat is None:
    dis_mat = sklearn.metrics.pairwise.euclidean_distances(np.mat(fts))
    #dis_mat = sklearn.metrics.pairwise.cosine_distances(np.mat(fts))
  num = len(lbls)

  for i in range(num):
    scores = dis_mat[:, i]
    targets = (lbls == lbls[i]).astype(np.uint8)
    sortind = np.argsort(scores, 0)   #[:top_k]
    truth = targets[sortind]

  num = len(models_ids)
  all_preds = []
  all_second_preds = []
  all_third_preds = []
  all_lbls = []
  all_scores = []
  all_mids = []
  visited = []
  for i in range(num):
    if i in visited:
      continue
    scores = fts[i, :]
    inds = (-scores).argsort()[:3]

    all_preds.append(inds[0])
    all_second_preds.append(inds[1])
    all_third_preds.append(inds[2])
    all_scores.append(scores)
    all_lbls.append(lbls[i])
    all_mids.append(models_ids[i])

  n_models = 0
  n_sucesses1 = 0
  n_sucesses2 = 0
  for i in range(len(all_mids)):
    gt = all_lbls[i]
    pred1 = all_preds[i]
    pred2 = all_second_preds[i]
    pred3 = all_third_preds[i]
    n_models += 1
    n_sucesses1 += (pred1 == gt)
    n_sucesses2 += (pred1 == gt or pred2==gt)# or pred3==gt)
  overall_accuracy1 = n_sucesses1 / n_models
  overall_accuracy3 = n_sucesses2 / n_models


  return {'Top1': overall_accuracy1, 'Top2': overall_accuracy3}

if __name__ == '__main__':

  base_path = 'runs/0364-12.11.2021..01.04__modelnet40_normal_resampled_not_random_walk'
  tst = np.load(os.path.join(base_path, 'features_3dfuture_2.npz'))
  mids = tst['model_ids']
  labels = tst['labels']
  feats = tst['features']
  #tst = calculate_map_auc(feats, labels, mids, logdir=base_path + '/results')
  tst = calculate_map_auc_by_classification(feats, labels, mids)#, logdir=base_path + '/results')
  #tst = calculate_Top2(feats, labels, mids)  # , logdir=base_path + '/results')

  print(tst)