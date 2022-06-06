import os, sys

import h5py
import trimesh
from easydict import EasyDict
import numpy as np
import open3d as o3d
import json

import utils


# Labels for all datasets
# -----------------------

model_net40_labels = [
  'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone'
  , 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard'
  , 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio'
  , 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase'
  , 'wardrobe', 'xbox'

]
model_net40_shape2label = {v: k for k, v in enumerate(model_net40_labels)}

model_net10_labels = [
  'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'

]
model_net10_shape2label = {v: k for k, v in enumerate(model_net10_labels)}

future3d_labels = [ 'Children Cabinet', 'Nightstand',
                    'Bookcase / jewelry Armoire', 'Wardrobe',
                    'Coffee Table', 'Corner/Side Table',
                    'Sideboard / Side Cabinet / Console table', 'Wine Cabinet',
                    'TV Stand', 'Drawer Chest / Corner cabinet',
                    'Shelf', 'Round End Table',
                    'King-size Bed', 'Bunk Bed',
                    'Bed Frame', 'Single bed',
                    'Kids Bed', 'Dining Chair',
                    'Lounge Chair / Cafe Chair / Office Chair', 'Dressing Chair',
                     'Classic Chinese Chair', 'Barstool',
                    'Dressing Table', 'Dining Table',
                    'Desk', 'Three-seat / Multi-seat Sofa',
                    'armchair', 'Loveseat Sofa',
                    'L-shaped Sofa', 'Lazy Sofa',
                    'Chaise Longue Sofa', 'Footstool / Sofastool / Bed End Stool / Stool',
                    'Pendant Lamp', 'Ceiling Lamp'
                  ]

future3d_shape2label = {v: k for k, v in enumerate(future3d_labels)}

scanobjectnn_labels = [
  'bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet'
]
scanobjectnn_shape2label = {v: k for k, v in enumerate(scanobjectnn_labels)}


def prepare_kdtree(point_cloud):
  vertices = point_cloud['vertices']
  point_cloud['kdtree_query'] = []
  t_mesh = trimesh.Trimesh(vertices=vertices, process=False)
  n_nbrs = min(21, vertices.shape[0] - 2)
  for n in range(vertices.shape[0]):
    d, i_nbrs = t_mesh.kdtree.query(vertices[n], n_nbrs)
    i_nbrs_cleared = [inbr for inbr in i_nbrs if inbr != n and inbr < vertices.shape[0]]
    if len(i_nbrs_cleared) > n_nbrs - 1:
      i_nbrs_cleared = i_nbrs_cleared[:n_nbrs - 1]
    point_cloud['kdtree_query'].append(np.array(i_nbrs_cleared, dtype=np.int32))
  point_cloud['kdtree_query'] = np.array(point_cloud['kdtree_query'])
  assert point_cloud['kdtree_query'].shape[1] == (n_nbrs - 1), 'Number of kdtree_query is wrong: ' + str(point_cloud['kdtree_query'].shape[1])


def add_fields_and_dump_model(data, fileds_needed, out_fn, dataset_name, dump_model=True):
  m = {}
  for k, v in data.items():
    if k in fileds_needed:
      m[k] = v
  for field in fileds_needed:
    if field not in m.keys():
      if field == 'labels':
        m[field] = np.zeros((0,))
      if field == 'dataset_name':
        m[field] = dataset_name
      if field == 'kdtree_query':
          prepare_kdtree(m)

  if dump_model:
    np.savez(out_fn, **m)

  return m

def pc_normalize(pc):
  l = pc.shape[0]
  centroid = np.mean(pc, axis=0)
  pc = pc - centroid
  m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
  pc = pc / m
  return pc

# ------------------------------------------------------- #

def prepare_modelnet40_normal_resampled():
  p = 'datasets_raw/modelnet40_normal_resampled/'
  p_out = 'datasets_processed-tmp/modelnet40_normal_resampled/'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)

  catfile = os.path.join(p, 'modelnet40_shape_names.txt')
  cat = [line.rstrip() for line in open(catfile)]
  classes = dict(zip(cat, range(len(cat))))
  fileds_needed = ['vertices', 'kdtree_query', 'label', 'dataset_name', 'vertex_normals']
  #npoints = 5000
  normalize = True
  for part in ['test', 'train']:
    npoints_list = [5000] if part == 'test' else [5000]
    for npoints in npoints_list:
        print('part: ', part)
        count_files = 0
        path_models_file_per_part = p + 'modelnet40_' + part + '.txt'
        files_name = [line.rstrip() for line in open(path_models_file_per_part)]
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in files_name]
        datapaths = [(files_name[i], shape_names[i], os.path.join(p, shape_names[i], files_name[i]) + '.txt') for i in range(len(files_name))]
        for f in datapaths:
          point_set = np.loadtxt(f[2], delimiter=',').astype(np.float32)
          cls = classes[f[1]]

          # Take the first npoints
          point_set = point_set[0:npoints, :]
          if normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

          file_name = os.path.basename(str(f))
          (prefix, sep, suffix) = file_name.rpartition('.')
          out_fn = p_out + part + '__' + str(npoints) + '__' + f[1] + '__' + prefix
          if os.path.isfile(out_fn + '.npz'):
            continue

          point_cloud_dict = EasyDict({'vertices': np.asarray(point_set[:, 0:3]), 'label': cls,
                                       'vertex_normals': np.asarray(point_set[:, 3:6])})
          add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)
          count_files += 1


def prepare_scanObjectNN():
  p = 'datasets_raw/scanObjectNN/'
  p_out = 'datasets_processed-tmp/scanObjectNN/'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  model_training_path = os.path.join(p, 'training_objectdataset_augmentedrot_scale75.h5')
  model_test_path = os.path.join(p, 'test_objectdataset_augmentedrot_scale75.h5')
  f_training = h5py.File(model_training_path, mode='a')
  f_test = h5py.File(model_test_path, mode='a')
  data_training = f_training['data'][:].astype('float32')
  label_training = f_training['label'][:].astype('int64')
  mask_training = f_training['mask'][:].astype('int64')
  data_test = f_test['data'][:].astype('float32')
  label_test = f_test['label'][:].astype('int64')
  mask_test = f_test['mask'][:].astype('int64')
  f_training.close()
  f_test.close()
  #unique, counts = np.unique(mask_training, return_counts=True)

  npoints = 2048
  fileds_needed = ['vertices', 'kdtree_query', 'label', 'dataset_name']
  for m in range(0, data_test.shape[0]):
    vertices = data_test[m, :, :]
    vertices = pc_normalize(vertices)
    label = label_test[m]
    mask = mask_test[m]
    mask[:] =1
    out_fn = p_out + 'test' + '__' + str(npoints) + '__' + str(m)
    point_cloud_dict = EasyDict({'vertices': vertices[:npoints], 'label': label})
    add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)
  for m in range(0, data_training.shape[0]):
    vertices = data_training[m, :, :]
    vertices = pc_normalize(vertices)
    label = label_training[m]
    mask = mask_training[m]
    mask[mask>0] = 0
    mask = mask+1
    out_fn = p_out + 'training' + '__' + str(npoints) + '__' + str(m)
    point_cloud_dict = EasyDict({'vertices': vertices[:npoints], 'label': label})
    add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)


def prepare_3D_FUTURE_resampled():
  p = 'datasets_raw/3DFutureModels/'
  p_out = 'datasets_processed-tmp/3DFutureModels/'
  if not os.path.isdir(p_out):
    os.makedirs(p_out)
  model_info_path = os.path.join(p, 'model_infos_from_scene.json')
  with open(model_info_path) as f:
    model_info = json.load(f)
  npoints_list = [1024]
  fileds_needed = ['vertices','kdtree_query', 'label', 'dataset_name', 'vertex_normals']
  for m in model_info:
    model_path = os.path.join(p, '3D-FUTURE-model/', m['model_id'], 'normalized_model.obj')
    tri_mesh = o3d.io.read_triangle_mesh(model_path)
    label = m['category']
    label_number = future3d_shape2label[label]
    if label == 'Dressing Chair' or label == 'Chaise Longue Sofa':
      continue
    part = 'train' if m['is_train']==True else 'test'

    for npoints in npoints_list:
      point_cloud_sampled_from_mesh = tri_mesh.sample_points_uniformly(number_of_points=npoints)
      vertices = np.asarray(point_cloud_sampled_from_mesh.points)

      out_fn = p_out + part + '__' + str(npoints) + '__' + m['model_id']
      if os.path.isfile(out_fn + '.npz') or (label == 'Dressing Chair') or (label == 'Chaise Longue Sofa'):
        continue

      point_cloud_dict = EasyDict({'vertices': vertices, 'label': label_number,
                                   'vertex_normals':  np.asarray(point_cloud_sampled_from_mesh.normals)})
      add_fields_and_dump_model(point_cloud_dict, fileds_needed, out_fn, dataset_name)

# ------------------------------------------------------- #

def prepare_one_dataset(dataset_name):
  dataset_name = dataset_name.lower()

  if dataset_name == 'modelnet40_normal_resampled':
    prepare_modelnet40_normal_resampled()

  if dataset_name == '3dfuture':
    prepare_3D_FUTURE_resampled()

  if dataset_name == 'scanobjectnn':
    prepare_scanObjectNN()


if __name__ == '__main__':
  utils.config_gpu(False)
  np.random.seed(1)

  if len(sys.argv) != 2:
    print('Use: python dataset_prepare.py <dataset name>')
    print('For example: python dataset_prepare.py modelnet40_normal_resampled')
    print('Another example: python dataset_prepare.py all')
  else:
    dataset_name = sys.argv[1]
    prepare_one_dataset(dataset_name)

