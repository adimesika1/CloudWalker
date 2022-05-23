from easydict import EasyDict
import numpy as np

import utils

def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
  n_vertices = mesh_extra['kdtree_query'].shape[0]#mesh_extra['n_vertices']
  kdtr = mesh_extra['kdtree_query']
  #vertices = mesh_extra['vertices']
  seq = np.zeros((seq_len + 1, ), dtype=np.int32)-1
  jumps = np.zeros((seq_len + 1,), dtype=np.bool)
  seq[0] = f0
  visited = np.zeros((n_vertices + 1,), dtype=np.bool)
  visited[-1] = True
  visited[f0] = True
  #sum_vertices = vertices[f0]
  for i in range(1, seq_len + 1):
    to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
    if len(to_consider):
      #random_or_not = np.random.rand(1)
      #if random_or_not<0:
      #  ind = np.argmax(np.sum(np.square(vertices[to_consider] - (vertices[to_consider] + sum_vertices) / (i + 1)), axis=1))
      #  seq[i] = to_consider[ind]
      #else:
      ind = np.random.choice(to_consider)
      seq[i] = ind
      jumps[i] = False
    else:
      seq[i] = np.random.randint(n_vertices)
      jumps[i] = True
      visited = np.zeros((n_vertices + 1,), dtype=np.bool)
      visited[-1] = True
    visited[seq[i]] = True
    #sum_vertices = sum_vertices + vertices[seq[i]]

  return seq, jumps


def get_model():
  from dataset_prepare import prepare_kdtree
  from dataset import load_model_from_npz

  model_fn = 'datasets_processed/modelnet40_normal_resampled/test__5000__toilet__toilet_0399.npz'
  model = load_model_from_npz(model_fn)
  model_dict = EasyDict({'vertices': np.asarray(model['vertices']), 'n_vertices': model['vertices'].shape[0],
                         'vertex_normals': np.asarray(model['vertex_normals'])})
  model_dict['vertices'] = model_dict['vertices'][0:1024]
  model_dict['vertex_normals'] = model_dict['vertex_normals'][0:1024]
  model_dict['n_vertices'] = 1024
  prepare_kdtree(model_dict)
  return model_dict


def show_walk_on_model(seed=0):
  walks = []
  vertices = model['vertices']
  coverage_value = np.array([0]*model['n_vertices'])
  for i in range(1):
    f0 = np.random.randint(model['n_vertices'])
    walk, jumps = get_seq_random_walk_local_jumps(model, f0, 500)
    coverage_value[walk] = 1
    walks.append(walk)
  #print("coverage value: ", np.count_nonzero(coverage_value)/model['Fn_vertices'])
  utils.visualize_model_walk(vertices, walks,seed=seed)


if __name__ == '__main__':
  utils.config_gpu(False)
  model = get_model()
  seed = 1969
  np.random.seed(seed)
  show_walk_on_model(seed)