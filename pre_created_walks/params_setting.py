from easydict import EasyDict
import numpy as np

import utils
import dataset_prepare

if 1:
  MAX_AUGMENTATION = 90
  run_folder = 'runs_test'

def set_up_default_params(network_task, run_name, cont_run_number=0):
  '''
  Define dafault parameters, commonly for many test case
  '''
  params = EasyDict()

  params.cont_run_number = cont_run_number
  params.run_root_path = 'runs'
  params.logdir = utils.get_run_folder(params.run_root_path + '/', '__' + run_name, params.cont_run_number)
  params.model_fn = params.logdir + '/learned_model.keras'
  params.ds_path = ''

  # Optimizer params
  params.optimizer_type = 'cycle'  # sgd / adam / cycle
  params.learning_rate_dynamics = 'cycle'
  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 1e-4,
                                    'step_size': 10000})
  params.n_models_per_test_epoch = 300
  params.gradient_clip_th = 1

  # Dataset params
  params.classes_indices_to_use = None
  params.train_dataset_size_limit = np.inf
  params.test_dataset_size_limit = np.inf
  params.network_task = network_task
  params.normalize_model = True
  params.datasets2use = {}
  params.test_data_augmentation = {}
  params.train_data_augmentation = {}

  params.additional_network_params = []
  params.use_vertex_normals = False
  params.n_walks_per_model = 1
  params.one_label_per_model = True
  params.train_loss = ['cros_entr']

  params.batch_size = int(32 / params.n_walks_per_model)
  params.num_members = 1


  # Other params
  params.log_freq = 100
  params.walk_alg = 'local_jumps'
  params.net_input = ['dxdydz']
  params.iter = 0

  params.net = 'RnnWalkNet'
  params.last_layer_activation = 'softmax'
  params.use_norm_layer = 'InstanceNorm'
  params.layer_sizes = None

  params.initializers = 'orthogonal'
  params.net_start_from_prev_net = None
  params.net_gru_dropout = 0

  params.full_accuracy_test = None
  params.best_local_acc = -1

  params.iters_to_train = 60e3

  return params

# Classifications
# ---------------

def modelnet10_normal_resampled_params():
  params = set_up_default_params('classification', 'modelnet10_normal_resampled', 0)
  params.net = 'RnnWalkNet'
  params.n_classes = 10
  params.use_vertex_normals = True
  params.last_layer_activation = 'softmax'

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 10000})

  params.train_data_augmentation = {'jitter': 0.1}

  ds_path ='datasets_processed/modelnet10_normal_resampled'
  params.datasets2use['train'] = [ds_path + '/*train*.npz']
  params.datasets2use['test'] = [ds_path + '/*test*.npz']


  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net10_labels,
                               'min_max_faces2use': params.test_min_max_faces2use,
                               'n_walks_per_model': 48 }

  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz', 'vertex_normals']
  params.walk_alg = 'local_jumps'

  # Set to start from prev net
  #path = 'runs/0329-21.10.2021..21.55__modelnet10_normal_resampled/learned_model2keep__00030008.keras'
  #params.net_start_from_prev_net = path

  return params


def modelnet40_normal_resampled_params():
  params = set_up_default_params('classification', 'modelnet40_normal_resampled', 0)
  params.net = 'RnnWalkNet'
  params.n_classes = 40
  params.last_layer_activation = 'softmax'
  params.use_vertex_normals = True

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 10000})
  params.train_data_augmentation = {'jitter': 0.1}

  params.ds_path ='../datasets_processed/modelnet40_normal_resampled'
  params.ds_path_dirs ='../datasets_processed/modelnet40_normal_resampled_dirs'
  params.datasets2use['train'] = [params.ds_path + '/*train*']
  params.datasets2use['test'] = [params.ds_path + '/*test*']

  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.model_net40_labels,
                               'n_models': np.inf,
                               'n_walks_per_model': 48 }

  # Parameters to recheck:
  params.iters_to_train = 200e3
  params.net_input = ['xyz', 'vertex_normals']
  params.walk_alg = 'local_jumps'

  # Set to start from prev net
  #path = 'runs/0286-25.02.2022..10.31__modelnet40_normal_resampled/learned_model2keep__00005219.keras'
  #params.net_start_from_prev_net = path

  return params


def scanobjectnn_params():
  params = set_up_default_params('classification', 'scanobjectnn', 0)
  params.n_classes = 15
  params.use_vertex_normals = False
  params.last_layer_activation = 'softmax'
  params.normalize_model = True

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 10000})

  ds_path ='../datasets_processed/scanObjectNN'
  params.ds_path_dirs ='../datasets_processed/scanObjectNN_dirs'
  params.datasets2use['train'] = [ds_path + '/*train*.npz']
  params.datasets2use['test'] = [ds_path + '/*test*.npz']

  params.seq_len = 800
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.scanobjectnn_labels,
                               'n_models': np.inf,
                               'n_walks_per_model': 48,#16 * 4,
                               }

  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['dxdydz']
  params.walk_alg = 'local_jumps'
  params.train_data_augmentation = {'translation': None, 'jitter': 0.1}

  # Set to start from prev net
  #path = 'runs/0020-19.12.2021..15.55__scanobjectnn/learned_model2keep__00080100.keras'
  #params.net_start_from_prev_net = path

  return params

def future3d_params():
  params = set_up_default_params('classification', 'future3d', 0)
  params.n_classes = 34
  params.use_vertex_normals = False
  params.last_layer_activation = 'softmax'
  params.use_saliency_weight = False

  params.cycle_opt_prms = EasyDict({'initial_learning_rate': 1e-6,
                                    'maximal_learning_rate': 0.0005,
                                    'step_size': 10000})

  params.ds_path ='../datasets_processed/3DFutureModels'
  params.ds_path_dirs ='../datasets_processed/3DFutureModels_dirs'
  params.datasets2use['train'] = [params.ds_path + '/*train*']
  params.datasets2use['test'] = [params.ds_path + '/*test*']

  params.seq_len = 600 #1400
  params.min_seq_len = int(params.seq_len / 2)

  params.full_accuracy_test = {'dataset_expansion': params.datasets2use['test'][0],
                               'labels': dataset_prepare.future3d_labels,
                               'n_models': np.inf,
                               'n_walks_per_model': 64,#16 * 4,
                               }

  # Parameters to recheck:
  params.iters_to_train = 500e3
  params.net_input = ['xyz']#, 'vertex_indices']
  params.walk_alg = 'local_jumps'
  params.train_data_augmentation = {'jitter': 0.1, 'translation': None}


  return params
