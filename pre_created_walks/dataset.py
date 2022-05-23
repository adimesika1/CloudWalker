import glob, os, copy
import random

import tensorflow as tf
import numpy as np

import utils
import walks

# Glabal list of dataset parameters. Used as part of runtime acceleration affort.
dataset_params_list = []

# ------------------------------------------------------------------ #
# ---------- Some utility functions -------------------------------- #
# ------------------------------------------------------------------ #
def load_model_from_npz(npz_fn):
  if npz_fn.find(':') != -1:
    npz_fn = npz_fn.split(':')[1]
  npz_fn = npz_fn.replace('\\', '/')
  mesh_data = np.load(npz_fn, encoding='latin1', allow_pickle=True)
  return mesh_data

def get_file_names(pathname_expansion, n_models):
  filenames = glob.glob(pathname_expansion)
  if n_models < len(filenames):
    filenames = filenames[0:n_models]
  assert len(filenames) > 0, 'DATASET error: no files in directory to be used! \nDataset directory: ' + pathname_expansion

  return filenames

def dump_all_fns_to_file(filenames, params):
  if 'logdir' in params.keys():
    for n in range(10):
      log_fn = params.logdir + '/dataset_files_' + str(n).zfill(2) + '.txt'
      if not os.path.isfile(log_fn):
        try:
          with open(log_fn, 'w') as f:
            for fn in filenames:
              f.write(fn + '\n')
        except:
          pass
        break


#For pre-processing
def norm_model(vertices):
  # Move the model so the bbox center will be at (0, 0, 0)
  mean = np.mean((np.min(vertices, axis=0), np.max(vertices, axis=0)), axis=0)
  vertices -= mean

  # Scale model to fit into the unit ball
  if 0: # Model Norm -->> !!!
    norm_with = np.max(vertices)
  else:
    norm_with = np.max(np.linalg.norm(vertices, axis=1))
  vertices *= 1/norm_with

  # Switch y and z dimensions
  vertices = vertices[:, [0, 2, 1]]

  #if norm_model.sub_mean_for_data_augmentation:
  #  vertices -= np.nanmean(vertices, axis=0)

def data_augmentation_rotation(vertices):
  max_rot_ang_deg = data_augmentation_rotation.max_rot_ang_deg
  x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
  A = np.array(((np.cos(x), -np.sin(x), 0),
                (np.sin(x), np.cos(x), 0),
                (0, 0, 1)),
               dtype=vertices.dtype)
  B = np.array(((np.cos(y), 0, -np.sin(y)),
                (0, 1, 0),
                (np.sin(y), 0, np.cos(y))),
               dtype=vertices.dtype)
  C = np.array(((1, 0, 0),
                (0, np.cos(z), -np.sin(z)),
                (0, np.sin(z), np.cos(z))),
               dtype=vertices.dtype)
  np.dot(vertices, A, out=vertices)
  np.dot(vertices, B, out=vertices)
  np.dot(vertices, C, out=vertices)

def jitter_point_cloud(vertices, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
      Input:
        Nx3 array, original point clouds
      Return:
        Nx3 array, jittered point clouds
  """
  clip = data_augmentation_rotation.clip
  N,C = vertices.shape
  assert (clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(N,C), -1 * clip, clip)
  jittered_data += vertices
  return jittered_data

def translate_point_cloud(batch_data, tval=0.2):
  """ Randomly translate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, translated batch of point clouds
  """
  n_batches = 1
  n_points = batch_data.shape[0]
  translation = np.random.uniform(-tval, tval, size=[n_batches, 3])
  translation = np.tile(np.expand_dims(translation, 1), [1, n_points, 1])
  batch_data = batch_data + translation

  return batch_data

def random_scale_point_cloud(data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
          Nx3 array, original point clouds
      Return:
          Nx3 array, scaled point clouds
  """
  N, C = data.shape
  scales = np.random.uniform(scale_low, scale_high)
  data[:, :] *= scales

  return data


# ------------------------------------------------------------------ #
# --- Some functions used to set up the RNN input "features" ------- #
# ------------------------------------------------------------------ #
def fill_xyz_features(features, f_idx, vertices, data_extra, seq, jumps, seq_len):
  walk = vertices[seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_dxdydz_features(features, f_idx, vertices, data_extra, seq, jumps, seq_len):
  walk = np.diff(vertices[seq[:seq_len + 1]], axis=0) * 100
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx


def fill_vertex_indices(features, f_idx, vertices, data_extra, seq, jumps, seq_len):
  walk = seq[1:seq_len + 1][:, None]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 1
  return f_idx

def fill_normals_features(features, f_idx, vertices, data_extra, seq, jumps, seq_len):
  walk = data_extra['vertex_normals'][seq[1:seq_len + 1]]
  features[:, f_idx:f_idx + walk.shape[1]] = walk
  f_idx += 3
  return f_idx

# ------------------------------------------------------------------ #
def setup_data_augmentation(dataset_params, data_augmentation):
  dataset_params.data_augmentaion_vertices_functions = []
  if 'rotation' in data_augmentation.keys() and data_augmentation['rotation']:
    data_augmentation_rotation.max_rot_ang_deg = data_augmentation['rotation']
    dataset_params.data_augmentaion_vertices_functions.append(data_augmentation_rotation)
  if 'jitter' in data_augmentation.keys():
    data_augmentation_rotation.clip = data_augmentation['jitter']
    dataset_params.data_augmentaion_vertices_functions.append(jitter_point_cloud)
  if 'translation' in data_augmentation.keys():
    dataset_params.data_augmentaion_vertices_functions.append(translate_point_cloud)
  if 'scale' in data_augmentation.keys():
    dataset_params.data_augmentaion_vertices_functions.append(random_scale_point_cloud)


def setup_features_params(dataset_params, params):
  dataset_params.fill_features_functions = []
  dataset_params.number_of_features = 0
  net_input = params.net_input
  if 'xyz' in net_input:
    dataset_params.fill_features_functions.append(fill_xyz_features)
    dataset_params.number_of_features += 3
  if 'dxdydz' in net_input:
    dataset_params.fill_features_functions.append(fill_dxdydz_features)
    dataset_params.number_of_features += 3
  if 'vertex_normals' in net_input:
    dataset_params.fill_features_functions.append(fill_normals_features)
    dataset_params.number_of_features += 3
  if 'vertex_indices' in net_input:
    dataset_params.fill_features_functions.append(fill_vertex_indices)
    dataset_params.number_of_features += 1

  elif params.walk_alg == 'local_jumps':
    dataset_params.walk_function = walks.get_seq_random_walk_local_jumps
    dataset_params.kdtree_query_needed = True
  else:
    raise Exception('Walk alg not recognized: ' + params.walk_alg)

  return dataset_params.number_of_features


# ------------------------------------------------- #
# ------- TensorFlow dataset functions ------------ #
# ------------------------------------------------- #
def generate_walk_py_fun(fn, vertices, kdtree_query, labels, params_idx, vertex_normals):
  return tf.py_function(
    generate_walk,
    inp=(fn, vertices, kdtree_query, labels, params_idx, vertex_normals),
    Tout=(fn.dtype, vertices.dtype, tf.int32)
  )


def generate_walk(fn, vertices,kdtree_query, labels_from_npz, params_idx, vertex_normals):
  mesh_data = {'vertices': vertices.numpy(),
               'kdtree_query': kdtree_query.numpy()}
  if dataset_params_list[params_idx[0]].label_per_step:
    mesh_data['labels'] = labels_from_npz.numpy()

  if dataset_params_list[params_idx[0]].use_vertex_normals:
    mesh_data['vertex_normals'] = vertex_normals.numpy()

  dataset_params = dataset_params_list[params_idx[0].numpy()]
  features, labels = point_cloud_data_to_walk_features(mesh_data, dataset_params)

  if dataset_params_list[params_idx[0]].label_per_step:
    labels_return = labels
  else:
    labels_return = labels_from_npz

  return fn[0], features, labels_return


def point_cloud_data_to_walk_features(data, dataset_params):
  vertices = data['vertices']
  seq_len = dataset_params.seq_len

  # Preprocessing
  if dataset_params.normalize_model:
    norm_model(vertices)

  # Data augmentation
  for data_augmentaion_function in dataset_params.data_augmentaion_vertices_functions:
    data_augmentaion_function(vertices)

  # Get essential data from file
  if dataset_params.label_per_step:
    mesh_labels = data['labels']
  else:
    mesh_labels = -1 * np.ones((vertices.shape[0],))

  data_extra = {}
  data_extra['n_vertices'] = vertices.shape[0]
  data_extra['vertices'] = data['vertices']
  if dataset_params.kdtree_query_needed:
    data_extra['kdtree_query'] = data['kdtree_query']
  if dataset_params.use_vertex_normals:
    data_extra['vertex_normals'] = data['vertex_normals']

  features = np.zeros((dataset_params.n_walks_per_model, seq_len, dataset_params.number_of_features), dtype=np.float32)
  labels   = np.zeros((dataset_params.n_walks_per_model, seq_len), dtype=np.int32)

  for walk_id in range(dataset_params.n_walks_per_model):
    f0 = np.random.randint(vertices.shape[0])
    seq, jumps = dataset_params.walk_function(data_extra, f0, seq_len)      # Get walk indices (and jump indications)
    f_idx = 0
    for fill_ftr_fun in dataset_params.fill_features_functions:
      f_idx = fill_ftr_fun(features[walk_id], f_idx, vertices, data_extra, seq, jumps, seq_len)
    if dataset_params.label_per_step:
      labels[walk_id] = mesh_labels[seq[1:seq_len + 1]]

  return features, labels


def setup_dataset_params(params, data_augmentation):
  p_idx = len(dataset_params_list)
  ds_params = copy.deepcopy(params)

  setup_data_augmentation(ds_params, data_augmentation)
  setup_features_params(ds_params, params)

  dataset_params_list.append(ds_params)

  return p_idx


class OpenMeshDataset(tf.data.Dataset):
  # OUTPUT:      (fn,               vertices,
  #               kdtree_query,    labels,
  #               params_idx,  vertex_normals)
  OUTPUT_SIGNATURE = (tf.TensorSpec(None, dtype=tf.string), tf.TensorSpec(None, dtype=tf.float32),
                      tf.TensorSpec(None, dtype=tf.int16), tf.TensorSpec(None, dtype=tf.int32),
                      tf.TensorSpec(None, dtype=tf.int16), tf.TensorSpec(None, dtype=tf.float32))

  def _generator(fn_, params_idx):
    fn = fn_[0]
    with np.load(fn, encoding='latin1', allow_pickle=True) as mesh_data:
      vertices = mesh_data['vertices']
      if dataset_params_list[params_idx].label_per_step:
        labels = mesh_data['labels']
      else:
        labels = mesh_data['label']
      if dataset_params_list[params_idx].kdtree_query_needed:
        kdtree_query = mesh_data['kdtree_query']
      else:
        kdtree_query = [-1]
      if dataset_params_list[params_idx].use_vertex_normals:
        vertex_normals = mesh_data['vertex_normals']  # .reshape([1,-1])
      else:
        vertex_normals = [-1]


      name = mesh_data['dataset_name'].tolist() + ':' + fn.decode()

    yield ([name], vertices, kdtree_query, labels, [params_idx], vertex_normals)

  def __new__(cls, filenames, params_idx):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_signature=cls.OUTPUT_SIGNATURE,
      args=(filenames, params_idx)
    )


def tf_point_cloud_dataset(params, pathname_expansion, size_limit=np.inf, shuffle_size=1000,
                           permute_file_names=True, n_models=np.inf, data_augmentation={}):
  params_idx = setup_dataset_params(params, data_augmentation)
  number_of_features = dataset_params_list[params_idx].number_of_features
  params.net_input_dim = number_of_features
  filenames = get_file_names(pathname_expansion, n_models)

  if permute_file_names:
    filenames = np.random.permutation(filenames)
  else:
    filenames.sort()
    filenames = np.array(filenames)

  if size_limit < len(filenames):
    filenames = filenames[:size_limit]
  n_items = len(filenames)
  dataset_params_list[params_idx].label_per_step = False
  dump_all_fns_to_file(filenames, params)

  def _open_npz_fn(*args):
    return OpenMeshDataset(args, params_idx)

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  if shuffle_size:
    ds = ds.shuffle(shuffle_size)
  ds = ds.interleave(_open_npz_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.map(generate_walk_py_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(params.batch_size, drop_remainder=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  #ds = ds.repeat()

  return ds, n_items

#for loading pre created walks
def pre_created_get_features_fun(fn, model_features, label):
  return tf.py_function(
    pre_created_get_features,
    inp=(fn, model_features, label),
    Tout=(fn.dtype, model_features.dtype, tf.int32)
  )

def pre_created_get_features(fn, model_features, label):
  return  fn[0], model_features, label


class pre_created_OpenFeaturesDataset(tf.data.Dataset):
  OUTPUT_SIGNATURE = (tf.TensorSpec(None, dtype=tf.string), tf.TensorSpec(None, dtype=tf.float32),
                      tf.TensorSpec(None, dtype=tf.int32))

  def _generator(fn_, params_idx):
    fn = fn_[0]
    npzs_in_dir = glob.glob(fn.decode("utf-8") +'*.npz')
    features = np.zeros((dataset_params_list[params_idx].n_walks_per_model, dataset_params_list[params_idx].seq_len,
                         dataset_params_list[params_idx].number_of_features), dtype=np.float32)
    chosen_random_file = np.random.choice(npzs_in_dir, dataset_params_list[params_idx].n_walks_per_model)
    for walk_id in range(len(chosen_random_file)):
      with np.load(chosen_random_file[walk_id], encoding='latin1', allow_pickle=True) as data:
        name = data['model_id']
        model_features = data['model_features']
        label = data['label']
        features[walk_id] = model_features


    yield ([name.tolist()], features, label)

  def __new__(cls, filenames, params_idx):
    return tf.data.Dataset.from_generator(
      cls._generator,
      output_signature=cls.OUTPUT_SIGNATURE,
      args=(filenames, params_idx)
    )


def tf_pre_created_dataset(params, pathname_expansion, size_limit=np.inf, shuffle_size=1000, permute_file_names=True):
  params_idx = len(dataset_params_list)
  ds_params = copy.deepcopy(params)
  dataset_params_list.append(ds_params)
  filenames = get_file_names_pre_created(pathname_expansion=pathname_expansion)

  if permute_file_names:
    filenames = np.random.permutation(filenames)
  else:
    filenames.sort()
    filenames = np.array(filenames)

  if size_limit < len(filenames):
    filenames = filenames[:size_limit]
  n_items = len(filenames)

  dump_all_fns_to_file(filenames, params)

  def _open_npz_fn(*args):
    return pre_created_OpenFeaturesDataset(args, params_idx)

  ds = tf.data.Dataset.from_tensor_slices(filenames)
  ds = ds.cache()
  if shuffle_size:
    ds = ds.shuffle(shuffle_size)
  ds = ds.interleave(_open_npz_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.map(pre_created_get_features_fun, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(params.batch_size, drop_remainder=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  return ds, n_items


def get_file_names_pre_created(pathname_expansion):
  return glob.glob(pathname_expansion)


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)
