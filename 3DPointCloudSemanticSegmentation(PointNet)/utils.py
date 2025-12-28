"""
Code written by Joey Wilson, 2023.
"""

import numpy as np
from sklearn.cluster import KMeans
import torch

import IPython
import plotly
import plotly.graph_objs as go


def hello():
  print("Welcome to assignment 4!")


def seed_torch():
  # torch.use_deterministic_algorithms(True)
  torch.manual_seed(808)
  np.random.seed(808)


COLOR_MAP = np.array(['#ffffff', '#f59664', '#f5e664', '#963c1e', '#b41e50',
                      '#ff0000', '#1e1eff', '#c828ff', '#5a1e96', '#ff00ff',
                      '#ff96ff', '#4b004b', '#4b00af', '#00c8ff', '#3278ff',
                      '#00af00', '#003c87', '#50f096', '#96f0ff', '#0000ff'])


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
        'size': 1,
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