"""
Code written by Joey Wilson, 2023.
Licensed under MIT License [see LICENSE].
"""

import numpy as np
import IPython
import plotly
import plotly.graph_objs as go
import cv2

COLOR_MAP = np.array(['#ffffff', '#f59664', '#f5e664', '#963c1e', '#b41e50',
                      '#ff0000', '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff',
                      '#ff96ff', '#4b004b', '#4b00af', '#00c8ff', '#3278ff',
                      '#00af00', '#003c87', '#50f096', '#96f0ff', '#0000ff'])

def hello():
  print("Welcome to assignment 3!")

def get_remap_lut(label_dict):
    '''
    remap_lut to remap classes of semantic kitti for training...
    :return:
    '''

    # make lookup table for mapping
    maxkey = max(label_dict.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(label_dict.keys())] = list(label_dict.values())

    return remap_lut

def configure_plotly_browser_state():
  display(IPython.core.display.HTML('''
      <script src="/static/components/requirejs/require.js"></script>
      <script>
        requirejs.config({
          paths: {
            base: '/static/base',
            plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
          },
        });
      </script>
      '''))

def plot_cloud(points, labels, max_num=100000):
  inds = np.arange(points.shape[0])
  inds = np.random.permutation(inds)[:max_num]
  points = points[inds, :]
  labels = labels[inds]

  trace = go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker={
        'size': 2,
        'opacity': 0.8,
        'color': COLOR_MAP[labels].tolist(),
    }
  )

  configure_plotly_browser_state()
  plotly.offline.init_notebook_mode(connected=False)

  layout = go.Layout(
      margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
      scene=dict(aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.2))
  )

  plotly.offline.iplot(go.Figure(data=[trace], layout=layout))

def get_init_mat(odometry, ind):
  init_mat = np.zeros((4, 4))
  init_mat[3, 3] = 1
  init_mat[:3, :4] = odometry[ind].reshape(3, 4)
  return init_mat


def get_cloud(velodyne_list, label_list, ind, label_map):
  xyz = np.fromfile(velodyne_list[ind],dtype=np.float32).reshape(-1,4)[:, :3]
  remap_lut = get_remap_lut(label_map)
  label = np.fromfile(label_list[ind], dtype=np.uint32).reshape(-1) & 0xFFFF
  return xyz, remap_lut[label]

def downsample_cloud(xyz, label, num):
  inds = np.arange(xyz.shape[0])
  inds = np.random.permutation(inds)[:num]
  points = xyz[inds, :]
  labels = label[inds]
  return points, labels


def depth_color(val, min_d=0, max_d=70):
  np.clip(val, 0, max_d, out=val)
  return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def print_projection_plt(image, points=None, color=None):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if points is not None:
      for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)