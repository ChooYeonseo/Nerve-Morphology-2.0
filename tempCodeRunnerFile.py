import numpy as np
import matplotlib.pyplot as plt
import os
import pprint as pprint
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree  # Import KDTree for faster nearest neighbor search



from nmutilis.bone_model import Bone
from nmutilis.NerveNodule import Nerve, Node
from nmutilis.readcsv import helper_dict, raw_dict



def resample_line(line, num_points=300):
    """Resample a 2D line to have evenly spaced points."""
    lengths = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
    cumulative_length = np.insert(np.cumsum(lengths), 0, 0)
    total_length = cumulative_length[-1]

    # Interpolation functions for x and y
    interp_x = interp1d(cumulative_length, line[:, 0], kind='linear')
    interp_y = interp1d(cumulative_length, line[:, 1], kind='linear')

    # Generate evenly spaced points along the total length
    new_lengths = np.linspace(0, total_length, num_points)
    resampled_line = np.stack((interp_x(new_lengths), interp_y(new_lengths)), axis=1)
    return resampled_line

bone = Bone(1, 2000)

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

nxlim, pxlim = (-5, 10)
nylim, pylim = (-5, 50)

x = np.linspace(nxlim, pxlim, 200)  # Grid range in x
y = np.linspace(nylim, pylim, 200)  # Grid range in y
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flattened grid for vectorized operations


# Plotting the heatmap
plt.figure(figsize=(8, 8))
plt.gca().set_aspect('equal')

# Iterate over the lines
heatmap = np.zeros_like(xx)
contribution_tracker = np.zeros_like(xx)

sigma = 0.35  # Small sigma for sharp falloff

for i in name_list:
    lines = branch_list[i]
    sorted_lines = sorted(lines, key=lambda line: np.max(line[:, 1]), reverse=True)
    temp = np.concatenate(sorted_lines, axis = 0)
    temp = resample_line(temp)
    distances = np.linalg.norm(grid_points[:, None, :] - temp[None, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    # Apply sharp Gaussian decay
    contribution = np.exp(-0.5 * (min_distances.reshape(xx.shape) / sigma) ** 2)
    heatmap += contribution
    contribution_tracker += 1

# Normalize the heatmap
heatmap = heatmap / np.maximum(contribution_tracker, 1)

# Plotting the heatmap

plt.contourf(xx, yy, heatmap, levels=100, cmap='hot', alpha=0.8)  # Heatmap
plt.plot(bone.x, bone.y, color="white")
plt.colorbar(label="Intensity")
plt.xlim(nxlim, pxlim)
plt.ylim(nylim, pylim)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Probabilistic Heatmap of Nerve with Branches")
plt.legend()
# plt.grid(True)
plt.show()