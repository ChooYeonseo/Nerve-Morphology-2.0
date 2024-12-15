import numpy as np
import matplotlib.pyplot as plt
import os
import pprint as pprint

from nmutilis.bone_model import Bone
from nmutilis.NerveNodule import Nerve, Node
from nmutilis.readcsv import helper_dict, raw_dict

from various_plots import (plot_heatmap, 
                           find_nth_maximum, 
                           plot_magnified_area, 
                           plot_magnified_area_with_bone, 
                           plot_sliced_graph,
                           plot_3d_sliced_graphs,
                           save_heatmap_mul)

avg_MM = 36.9
scale = 1 * avg_MM / 5

bone = Bone(scale, 2000)

nxlim, pxlim = [x * scale for x in [-5, 10]]
nylim, pylim = [x * scale for x in [-5, 50]]
lims = [nxlim, pxlim, nylim, pylim]
sigma = 3  # Small sigma for sharp falloff
n = 1
# area_bounds = [2, 8, 11, 20] # MID10MID12
area_bounds = [-2, 7, -4, 7] # MID10MID12
area_bounds =[x * scale for x in area_bounds]
y_level = 0
y_levels = [0, 2, 4, 6, 8]

folder_path = "./branches"
branches = [file for file in os.listdir(folder_path) if file.endswith('.npz')]
branch_list = {}
name_list = []
print("NPZ Files:", branches)

for branch in branches[:]:
    dir = folder_path + "/"+ branch
    name = branch.split("_")[0]
    name_list.append(name)
    temp_npz = np.load(dir)
    li = []
    for i in temp_npz:
        li.append(temp_npz[i])
    branch_list[name] = li

# Generate a heatmap of the nerve branches
heatmap, xx, yy = plot_heatmap(lims, bone, sigma, branch_list, name_list)
# save_heatmap_mul(lims, bone, sigma, branch_list, name_list)
find_nth_maximum(1, heatmap, xx, yy, area_bounds, bone, lims, scale)
# plot_magnified_area(1, heatmap, xx, yy, area_bounds)
plot_magnified_area_with_bone(1, heatmap, xx, yy, area_bounds, bone, scale)
plot_sliced_graph(y_level, heatmap, xx, yy)
# plot_3d_sliced_graphs(y_levels, heatmap, xx, yy)