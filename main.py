import numpy as np
import matplotlib.pyplot as plt
import os
import pprint as pprint

from nmutilis.bone_model import Bone
from nmutilis.NerveNodule import Nerve, Node
from nmutilis.readcsv import helper_dict, raw_dict

from various_plots import plot_heatmap, find_nth_maximum, plot_magnified_area, plot_magnified_area_with_bone

bone = Bone(1, 2000)
nxlim, pxlim = (-5, 10)
nylim, pylim = (-5, 50)
lims = [nxlim, pxlim, nylim, pylim]
sigma = 0.35  # Small sigma for sharp falloff
n = 1
area_bounds = [-2, 7, -4, 5] 

folder_path = "./branches"
branches = [file for file in os.listdir(folder_path) if file.endswith('.npz')]
branch_list = {}
name_list = []
print("NPZ Files:", branches)

for branch in branches:
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
find_nth_maximum(1, heatmap, xx, yy, area_bounds, bone, lims)
plot_magnified_area(1, heatmap, xx, yy, area_bounds)
plot_magnified_area_with_bone(1, heatmap, xx, yy, area_bounds, bone)