import os, shutil, psutil, json, copy
import datetime
import pylab as plt
import numpy as np
import tensorflow as tf
import pyvista as pv

import evaluate_classification

class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

def config_gpu(use_gpu=True):
  print('tf.__version__', tf.__version__)
  np.set_printoptions(suppress=True)
  #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  try:
    if use_gpu:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpus, True)
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  except:
    pass

def get_gpu_temprature():
  #output = os.popen("nvidia-smi -q | grep 'GPU Current Temp' | cut -d' ' -f 24").read()
  #output = ''.join(filter(str.isdigit, output))
  #try:
  #  temp = int(output)
  #except:
  #  temp = 0
  return 0#temp


def backup_python_files_and_params(params):
  save_id = 0
  while 1:
    code_log_folder = params.logdir + '/.' + str(save_id)
    if not os.path.isdir(code_log_folder):
      os.makedirs(code_log_folder)
      for file in os.listdir():
        if file.endswith('py'):
          shutil.copyfile(file, code_log_folder + '/' + file)
      break
    else:
      save_id += 1

  # Dump params to text file
  try:
    prm2dump = copy.deepcopy(params)
    if 'hyper_params' in prm2dump.keys():
      prm2dump.hyper_params = str(prm2dump.hyper_params)
      prm2dump.hparams_metrics = prm2dump.hparams_metrics[0]._display_name
      for l in prm2dump.net:
        l['layer_function'] = 'layer_function'
    with open(params.logdir + '/params.txt', 'w') as fp:
      json.dump(prm2dump, fp, indent=2, sort_keys=True)
  except:
    pass


def get_run_folder(root_dir, str2add='', cont_run_number=False):
  try:
    all_runs = os.listdir(root_dir)
    run_ids = [int(d.split('-')[0]) for d in all_runs if '-' in d]
    if cont_run_number:
      n = [i for i, m in enumerate(run_ids) if m == cont_run_number][0]
      run_dir = root_dir + all_runs[n]
      print('Continue to run at:', run_dir)
      return run_dir
    n = np.sort(run_ids)[-1]
  except:
    n = 0
  now = datetime.datetime.now()
  return root_dir + str(n + 1).zfill(4) + '-' + now.strftime("%d.%m.%Y..%H.%M") + str2add


last_free_mem = np.inf
def check_mem_and_exit_if_full():
  global last_free_mem
  free_mem = psutil.virtual_memory().available + psutil.swap_memory().free
  free_mem_gb = round(free_mem / 1024 / 1024 / 1024, 2)
  if last_free_mem > free_mem_gb + 0.25:
    last_free_mem = free_mem_gb
    print('free_mem', free_mem_gb, 'GB')
  if free_mem_gb < 1:
    print('!!! Exiting due to memory full !!!')
    exit(111)
  return free_mem_gb


def visualize_model_walk(vertices, walks, title='', cpos=None, color='black',edge_color =None,seed=0):
  #set lights
  light = pv.Light(position=(2.3, 1.3, -4), color='white', light_type="headlight")#, intensity=1)
  light1 = pv.Light(position=(-1, 2, 1), color='gray')

  #add all vertices
  surf = pv.PolyData(vertices)
  p = pv.Plotter(lighting='none', window_size=[1200,1200])
  p.set_background('white')
  p.add_mesh(pv.PolyData(surf.points), point_size=6, render_points_as_spheres=True, color=[0.80,0.80,0.80])

  if edge_color is None:
    cm = np.array(plt.get_cmap('plasma',len(walks[0])).colors)
    edge_color = cm[:,:]
  if type(walks) is list:
    for wi in range(len(walks)):
      walk = list(map(int, walks[wi]))

      #render walk edges
      all_edges = [[2, walk[i], walk[i + 1]] for i in range(len(walk) - 1)]
      for i, edge in enumerate(all_edges):
        walk_edges = np.array(edge)
        walk_mesh = pv.PolyData(vertices, walk_edges)
        color_edges = list(edge_color[i])
        p.add_mesh(walk_mesh, show_edges=True, line_width=6, edge_color=color_edges, render_lines_as_tubes=True)

      #size for points
      for i, c in zip(walk, edge_color):
        if i == walk[0]:
          point_size = 10
        elif i == walk[-1]:
          point_size = 6
        else:
          point_size = 8
        p.add_mesh(pv.PolyData(surf.points[i]), color=list(c), point_size=point_size, render_points_as_spheres=True, opacity=1, smooth_shading=True)

  #set camera
  p.camera_position= [(-4.011262534048553, 1.275820047868118, -3.8828205787805063), (0, 0, 0), (0, 1, 0)]
  p.camera.zoom(1.5)

  p.add_light(light)
  p.add_light(light1)
  p.show()#screenshot='images/toiletCW'+str(seed)+'.png')
  print(p.camera.position)
  print(p.camera.view_angle)

next_iter_to_keep = 0 # Should be set by -train_val- function, each time job starts
def save_model_if_needed(iterations, dnn_model, params):
  global next_iter_to_keep
  iter_th = 10000
  keep = iterations.numpy() >= next_iter_to_keep
  dnn_model.save_weights(params.logdir, iterations.numpy(), keep=keep)
  if keep:
    if iterations < iter_th:
      next_iter_to_keep = iterations * 2
    else:
      next_iter_to_keep = int(iterations / iter_th) * iter_th + iter_th
    if params.full_accuracy_test is not None:
      if params.network_task == 'classification':
        accuracy, _ = evaluate_classification.calc_accuracy_test(params=params, dnn_model=dnn_model, **params.full_accuracy_test) #Adi - need to change
        with open(params.logdir + '/log.txt', 'at') as f:
          f.write('Accuracy: ' + str(np.round(np.array(accuracy) * 100, 3)) + '%, Iter: ' + str(iterations.numpy()) + '\n')
        tf.summary.scalar('full_accuracy_test/overall', accuracy[0], step=iterations)
        tf.summary.scalar('full_accuracy_test/mean', accuracy[1], step=iterations)

def get_dataset_type_from_name(tf_names):
  name_str = tf_names[0].numpy().decode()
  return name_str[:name_str.find(':')]

def get_model_name_from_npz_fn(npz_fn):
  fn = npz_fn.split('/')[-1].split('.')[-2]
  sp_fn = fn.split('\\')[0]
  model_name = sp_fn

  return model_name
