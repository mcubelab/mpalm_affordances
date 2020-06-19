import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import axes3d
from plotly.subplots import make_subplots


def plot_pointcloud(data, mask):
    red_marker = {
        'size' : 1.0,
        'color' : 'red',                # set color to an array/list of desired values
        'colorscale' : 'Viridis',   # choose a colorscale
        'opacity' : 0.5
    }
    blue_marker = {
        'size' : 3.0,
        'color' : 'blue',                # set color to an array/list of desired values
        'colorscale' : 'Viridis',   # choose a colorscale
        'opacity' : 1.0
    }
    black_marker = {
        'size' : 3.0,
        'color' : 'black',                # set color to an array/list of desired values
        'colorscale' : 'Viridis',   # choose a colorscale
        'opacity' : 0.8
    }
    pcd_start = data
    pcd_start_masked = data[mask > 0.3, :]

    start_pcd_data = {
        'type': 'scatter3d',
        'x': pcd_start[:, 0],
        'y': pcd_start[:, 1],
        'z': pcd_start[:, 2],
        'mode': 'markers',
        'marker': red_marker
    }

    start_mask_data = {
        'type': 'scatter3d',
        'x': pcd_start_masked[:, 0],
        'y': pcd_start_masked[:, 1],
        'z': pcd_start_masked[:, 2],
        'mode': 'markers',
        'marker': blue_marker
    }
    plane_data = {
        'type': 'mesh3d',
        'x': [-1, 1, 1, -1],
        'y': [-1, -1, 1, 1],
        'z': [0, 0, 0, 0],
        'color': 'gray',
        'opacity': 0.5,
        'delaunayaxis': 'z'
    }
    fig_data = []
    fig_data.append(plane_data)
    fig_data.append(start_pcd_data)
    fig_data.append(start_mask_data)
    fig = go.Figure(data=fig_data)
    camera = {
        'up': {'x': 0, 'y': 0,'z': 1},
        'center': {'x': 0.45, 'y': 0, 'z': 0.0},
        'eye': {'x': -1.0, 'y': 0.0, 'z': 0.01}
    }
    scene = {
        'xaxis': {'nticks': 10, 'range': [-0.1, 0.9]},
        'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
        'zaxis': {'nticks': 8, 'range': [-0.01, 0.99]}
    }
    width = 700
    margin = {'r': 20, 'l': 10, 'b': 10, 't': 10}
    fig.update_layout(
        scene=scene,
        scene_camera=camera,
        width=width,
        margin=margin
    )
    return fig

def main():
    np_data = np.load('/home/anthony/repos/research/mpalm_affordances/catkin_ws/src/primitives/data/grasp/numpy_grasp_table_pcd/0_1124.npz')
    fig = plot_pointcloud(np_data)
    fig.show()
    fig_ds = plot_pointcloud(np_data, downsampled=True)
    fig_ds.show()

if __name__ == "__main__":
    main()
