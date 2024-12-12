import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree  # Import KDTree for faster nearest neighbor search
from tqdm import tqdm
import pprint as pprint
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter



# def resample_line(line, num_points=300):
#     """Resample a 2D line to have evenly spaced points."""
#     lengths = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
#     cumulative_length = np.insert(np.cumsum(lengths), 0, 0)
#     total_length = cumulative_length[-1]

#     # Interpolation functions for x and y
#     interp_x = interp1d(cumulative_length, line[:, 0], kind='linear')
#     interp_y = interp1d(cumulative_length, line[:, 1], kind='linear')

#     # Generate evenly spaced points along the total length
#     new_lengths = np.linspace(0, total_length, num_points)
#     resampled_line = np.stack((interp_x(new_lengths), interp_y(new_lengths)), axis=1)
#     return resampled_line

import numpy as np
from scipy.interpolate import interp1d

def resample_line(line, density=500):
    """
    Resample a 2D line to have points spaced according to a given density.
    
    Parameters:
        line (ndarray): A 2D array of shape (N, 2) representing the line coordinates.
        density (float): Desired number of points per unit length.
        
    Returns:
        resampled_line (ndarray): A 2D array of resampled points.
    """
    # Calculate segment lengths
    lengths = np.sqrt(np.sum(np.diff(line, axis=0)**2, axis=1))
    cumulative_length = np.insert(np.cumsum(lengths), 0, 0)
    total_length = cumulative_length[-1]

    # Determine the total number of points based on the density
    num_points = int(np.ceil(total_length * density))

    # Interpolation functions for x and y
    interp_x = interp1d(cumulative_length, line[:, 0], kind='linear')
    interp_y = interp1d(cumulative_length, line[:, 1], kind='linear')

    # Generate evenly spaced lengths along the total length
    new_lengths = np.linspace(0, total_length, num_points)
    resampled_line = np.stack((interp_x(new_lengths), interp_y(new_lengths)), axis=1)
    print(np.shape(resampled_line))
    
    return resampled_line

def plot_heatmap(lims, bone, sigma, branch_list, name_list, show=True):
    """
    Plot a heatmap of the nerve with branches.
    
    Parameters:
        lims (list): Limits for the heatmap as [xmin, xmax, ymin, ymax].
        bone (Bone): Bone object containing its shape data.
        sigma (float): Standard deviation for the Gaussian kernel.
        branch_list (dict): Dictionary of branch names and their line segments.
    """
    nxlim, pxlim, nylim, pylim = lims
    x = np.linspace(nxlim, pxlim, 200)  # Grid range in x
    y = np.linspace(nylim, pylim, 200)  # Grid range in y
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flattened grid for vectorized operations

    heatmap = np.zeros_like(xx)

    for i in name_list:
        lines = branch_list[i]  # Get the list of sub-branches for this branch
        for line in lines:
            # Resample the current sub-branch line
            resampled_line = resample_line(line, density=600)

            x_indices, y_indices = resampled_line.T
            grid_x_indices = np.clip(((x_indices[1:] - nxlim) / (pxlim - nxlim) * (xx.shape[1] - 1)).astype(int), 0, xx.shape[1] - 1)
            grid_y_indices = np.clip(((y_indices[1:] - nylim) / (pylim - nylim) * (xx.shape[0] - 1)).astype(int), 0, xx.shape[0] - 1)
            heatmap[grid_y_indices, grid_x_indices] += 1
            # For each segment of the resampled line, mark the grid
            # for j in range(len(resampled_line) - 1):
            #     x0, y0 = resampled_line[j]
            #     x1, y1 = resampled_line[j + 1]

            #     # Find the indices of the grid points nearest to the line segment
            #     x_indices = np.linspace(x0, x1, 100)  # Increase 100 for smoother raster
            #     y_indices = np.linspace(y0, y1, 100)
            #     grid_x_indices = np.clip(((x_indices - nxlim) / (pxlim - nxlim) * (xx.shape[1] - 1)).astype(int), 0, xx.shape[1] - 1)
            #     grid_y_indices = np.clip(((y_indices - nylim) / (pylim - nylim) * (xx.shape[0] - 1)).astype(int), 0, xx.shape[0] - 1)

            #     # Mark the grid points
            #     heatmap[grid_y_indices, grid_x_indices] += 1

    # Smooth the grid with a Gaussian filter
    heatmap /= heatmap.max()
    k = 10  # Steepness of the sigmoid
    x0 = 0.2  # Center of the sigmoid
    heatmap = 1 / (1 + np.exp(-k * (heatmap - x0)))
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Plotting the heatmap
    if show:
        plt.figure(figsize=(5, 8))
        plt.gca().set_aspect('equal')
        plt.contourf(xx, yy, heatmap, levels=100, cmap='hot', alpha=0.8)  # Heatmap
        plt.plot(bone.x, bone.y, color="white")
        plt.colorbar(label="Intensity")
        plt.xlim(nxlim, pxlim)
        plt.ylim(nylim, pylim)
        plt.xticks([nxlim, pxlim], ['Anterior', 'Posterior'])
        plt.yticks([nylim, pylim], ['Inferior', 'Superior'])
        plt.title("Probabilistic Heatmap of Nerve with Branches")
        plt.grid(True)
        plt.show()
    return heatmap, xx, yy

def save_heatmap_mul(lims, bone, sigma, branch_list, name_list):

    nxlim, pxlim, nylim, pylim = lims
    for name in name_list:
        temp_dict = {}
        temp_dict[name] = branch_list[name]

        heatmap, xx, yy = plot_heatmap(lims, bone, sigma, temp_dict, [name], show=False)
        plt.figure(figsize=(4, 8))
        plt.gca().set_aspect('equal')
        plt.contourf(xx, yy, heatmap, levels=100, cmap='hot', alpha=0.8)
        plt.plot(bone.x, bone.y, color="white")
        plt.colorbar(label="Intensity")
        plt.xlim(nxlim, pxlim)
        plt.ylim(nylim, pylim)
        plt.xticks([nxlim, pxlim], ['Anterior', 'Posterior'])
        plt.yticks([nylim, pylim], ['Inferior', 'Superior'])
        plt.grid(True)
        plt.title("{} Heatmap".format(name))
        special_coord = [bone.get_mid10()[1], bone.get_mid12()[1], bone.get_MID(), bone.get_TIP(), bone.get_LAT()]
        landmark_name = ['MID10', 'MID12', 'MID', 'TIP', 'LAT']
        for i, co in enumerate(special_coord):
            print(co)
            plt.scatter(co[0], co[1], color='blue', s=20, zorder=5)
            if landmark_name[i] == 'MID':
                plt.text(co[0] - 3.5, co[1], landmark_name[i], fontsize=12, zorder=6, color='white')
            elif landmark_name[i] == 'TIP':
                plt.text(co[0], co[1] - 1.5, landmark_name[i], fontsize=12, zorder=6, color='white')
            else:
                plt.text(co[0] + 0.5, co[1], landmark_name[i], fontsize=12, zorder=6, color='white')
        plt.subplots_adjust(left=0.05, right=0.85)  # Adjust values for alignment
        plt.savefig('./SN/{}_heatmap.png'.format(name))

def find_nth_maximum(n, heatmap, xx, yy, bounds, bone, lims):
    """
    Find the nth maximum point within a specified area of the heatmap.
    
    Parameters:
        n (int): The rank of the maximum point (e.g., 1 for max, 2 for 2nd max).
        heatmap (numpy array): The heatmap array.
        xx, yy (numpy array): Grid arrays for x and y coordinates.
        bounds (list): Area bounds as [xmin, xmax, ymin, ymax].
        
    Returns:
        tuple: Coordinates (x, y) and intensity value of the nth maximum point.
    """
    xmin, xmax, ymin, ymax = bounds
    nxlim, pxlim, nylim, pylim = lims
    # Find the indices within the specified area
    x_indices = (xx[0, :] >= xmin) & (xx[0, :] <= xmax)
    y_indices = (yy[:, 0] >= ymin) & (yy[:, 0] <= ymax)
    sub_heatmap = heatmap[np.ix_(y_indices, x_indices)]
    sub_xx = xx[np.ix_(y_indices, x_indices)]
    sub_yy = yy[np.ix_(y_indices, x_indices)]
    
    # Flatten the subregion for sorting
    flat_heatmap = sub_heatmap.ravel()
    flat_x = sub_xx.ravel()
    flat_y = sub_yy.ravel()

    # Get sorted indices in descending order
    sorted_indices = np.argsort(flat_heatmap)[::-1]

    if n > len(sorted_indices):
        print(f"Error: There are only {len(sorted_indices)} points in the area.")
        return None

    # Get the nth maximum value and its coordinates
    nth_idx = sorted_indices[n - 1]
    nth_max_value = flat_heatmap[nth_idx]
    nth_max_x = flat_x[nth_idx]
    nth_max_y = flat_y[nth_idx]

    if nth_max_x is not None:
        print(f"The maximum intensity is {nth_max_value} at ({nth_max_x}, {nth_max_y})")

        # Plot the heatmap with the nth maximum point
        plt.figure(figsize=(5, 8))
        plt.gca().set_aspect('equal')
        plt.contourf(xx, yy, heatmap, levels=100, cmap='hot', alpha=0.8)  # Heatmap
        plt.colorbar(label="Intensity")
        plt.plot(bone.x, bone.y, color="white")


        # Plot the specified area
        plt.plot(
            [bounds[0], bounds[1], bounds[1], bounds[0], bounds[0]],
            [bounds[2], bounds[2], bounds[3], bounds[3], bounds[2]],
            color="cyan",
            linestyle="--",
            label="Specified Area"
        )

        # Plot the nth maximum intensity point
        plt.scatter(nth_max_x, nth_max_y, color="blue", s=50, zorder=5, label=f"Max Point")
        plt.annotate(f"Max ({nth_max_x:.2f}, {nth_max_y:.2f})",
                     (nth_max_x, nth_max_y),
                     textcoords="offset points", xytext=(-20, 10),
                     color="blue", fontsize=10, arrowprops=dict(arrowstyle="->", color="blue"))


        special_coord = [bone.get_mid10()[1], bone.get_mid12()[1], bone.get_MID(), bone.get_TIP(), bone.get_LAT()]
        name = ['MID10', 'MID12', 'MID', 'TIP', 'LAT']
        for i, co in enumerate(special_coord):
            print(co)
            plt.scatter(co[0], co[1], color='blue', s=20, zorder=5)
            if name[i] == 'MID':
                plt.text(co[0] - 3.5, co[1], name[i], fontsize=12, zorder=6, color='white')
            elif name[i] == 'TIP':
                plt.text(co[0], co[1] - 1.5, name[i], fontsize=12, zorder=6, color='white')
            else:
                plt.text(co[0] + 0.5, co[1], name[i], fontsize=12, zorder=6, color='white')



        plt.xlim(nxlim, pxlim)
        plt.ylim(nylim, pylim)
        plt.title("Probabilistic Heatmap of Nerve with Branches")
        plt.legend()
        plt.grid(True)
        plt.show()

    return nth_max_x, nth_max_y, nth_max_value

def plot_magnified_area(n, heatmap, xx, yy, bounds):
    """
    Plot a magnified view of a specified area and mark the nth maximum point.
    
    Parameters:
        n (int): The rank of the maximum point (e.g., 1 for max, 2 for 2nd max).
        heatmap (numpy array): The heatmap array.
        xx, yy (numpy array): Grid arrays for x and y coordinates.
        bounds (list): Area bounds as [xmin, xmax, ymin, ymax].
    """
    xmin, xmax, ymin, ymax = bounds

    # Find the indices within the specified area
    x_indices = (xx[0, :] >= xmin) & (xx[0, :] <= xmax)
    y_indices = (yy[:, 0] >= ymin) & (yy[:, 0] <= ymax)
    sub_heatmap = heatmap[np.ix_(y_indices, x_indices)]
    sub_xx = xx[np.ix_(y_indices, x_indices)]
    sub_yy = yy[np.ix_(y_indices, x_indices)]
    
    # Flatten the subregion for sorting
    flat_heatmap = sub_heatmap.ravel()
    flat_x = sub_xx.ravel()
    flat_y = sub_yy.ravel()

    # Get sorted indices in descending order
    sorted_indices = np.argsort(flat_heatmap)[::-1]

    if n > len(sorted_indices):
        print(f"Error: There are only {len(sorted_indices)} points in the area.")
        return None

    # Get the nth maximum value and its coordinates
    nth_idx = sorted_indices[n - 1]
    nth_max_value = flat_heatmap[nth_idx]
    nth_max_x = flat_x[nth_idx]
    nth_max_y = flat_y[nth_idx]

    # Plot the magnified view
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')
    plt.contourf(sub_xx, sub_yy, sub_heatmap, levels=100, cmap='hot', alpha=0.8)  # Heatmap
    plt.colorbar(label="Intensity")

    # Plot the nth maximum intensity point
    plt.scatter(nth_max_x, nth_max_y, color="blue", s=50, zorder=5, label=f"{n}th Max Point")
    plt.annotate(f"{n}th Max ({nth_max_x:.2f}, {nth_max_y:.2f})",
                 (nth_max_x, nth_max_y),
                 textcoords="offset points", xytext=(-20, 10),
                 color="blue", fontsize=10, arrowprops=dict(arrowstyle="->", color="blue"))

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(f"Magnified View of Specified Area with {n}th Max Point")
    plt.legend()
    plt.grid(True)
    plt.show()

    return nth_max_x, nth_max_y, nth_max_value

def plot_magnified_area_with_bone(n, heatmap, xx, yy, bounds, bone):
    """
    Plot a magnified view of a specified area, mark the nth maximum point, 
    and display the section of the bone within the magnified area.
    
    Parameters:
        n (int): The rank of the maximum point (e.g., 1 for max, 2 for 2nd max).
        heatmap (numpy array): The heatmap array.
        xx, yy (numpy array): Grid arrays for x and y coordinates.
        bounds (list): Area bounds as [xmin, xmax, ymin, ymax].
        bone (Bone): Bone object containing its shape data.
    """
    xmin, xmax, ymin, ymax = bounds

    # Find the indices within the specified area
    x_indices = (xx[0, :] >= xmin) & (xx[0, :] <= xmax)
    y_indices = (yy[:, 0] >= ymin) & (yy[:, 0] <= ymax)
    sub_heatmap = heatmap[np.ix_(y_indices, x_indices)]
    sub_xx = xx[np.ix_(y_indices, x_indices)]
    sub_yy = yy[np.ix_(y_indices, x_indices)]
    
    # Flatten the subregion for sorting
    flat_heatmap = sub_heatmap.ravel()
    flat_x = sub_xx.ravel()
    flat_y = sub_yy.ravel()

    # Get sorted indices in descending order
    sorted_indices = np.argsort(flat_heatmap)[::-1]

    if n > len(sorted_indices):
        print(f"Error: There are only {len(sorted_indices)} points in the area.")
        return None

    # Get the nth maximum value and its coordinates
    nth_idx = sorted_indices[n - 1]
    nth_max_value = flat_heatmap[nth_idx]
    nth_max_x = flat_x[nth_idx]
    nth_max_y = flat_y[nth_idx]

    # Filter the bone points to include only those within the bounds
    bone_points = np.array([bone.x, bone.y]).T
    filtered_bone_points = bone_points[
        (bone_points[:, 0] >= xmin) & (bone_points[:, 0] <= xmax) &
        (bone_points[:, 1] >= ymin) & (bone_points[:, 1] <= ymax)
    ]

    # Plot the magnified view
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')
    plt.contourf(sub_xx, sub_yy, sub_heatmap, levels=100, cmap='hot', alpha=0.8)  # Heatmap
    plt.colorbar(label="Intensity")

    # Plot the filtered bone points
    if filtered_bone_points.size > 0:
        plt.plot(filtered_bone_points[:, 0], filtered_bone_points[:, 1], color="white", linewidth=2, label="Bone")

    # Plot the nth maximum intensity point
    plt.scatter(nth_max_x, nth_max_y, color="blue", s=50, zorder=5, label=f"Max Point")
    plt.annotate(f"Max ({nth_max_x:.2f}, {nth_max_y:.2f})",
                 (nth_max_x, nth_max_y),
                 textcoords="offset points", xytext=(-20, 10),
                 color="blue", fontsize=10, arrowprops=dict(arrowstyle="->", color="blue"))

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(f"Magnified View of Specified Area with Max Point and Bone Section")
    plt.legend()
    plt.grid(True)
    plt.show()

    return nth_max_x, nth_max_y, nth_max_value

def plot_sliced_graph(y_value, heatmap, xx, yy):
    """
    Plot a sliced graph of intensity vs. x-axis values at a specific y-axis level.

    Parameters:
        y_value (float): The y-coordinate at which to slice the heatmap.
        heatmap (numpy array): The heatmap array.
        xx, yy (numpy array): Grid arrays for x and y coordinates.
    """
    # Find the closest index to the provided y_value
    y_index = np.argmin(np.abs(yy[:, 0] - y_value))

    # Extract the intensity slice along the closest y-axis row
    intensity_slice = heatmap[y_index, :]
    x_values = xx[y_index, :]

    # Plot the sliced graph
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, intensity_slice, label=f"Sliced Graph at y={y_value:.2f}", color="red", lw=2)
    plt.title(f"Intensity vs. x-axis at y={y_value:.2f}")
    plt.xlabel("x-axis")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.show()

    return x_values, intensity_slice

def plot_3d_sliced_graphs(y_values, heatmap, xx, yy):
    """
    Plot a 3D graph of intensity slices along the x-axis for a list of y-axis values.

    Parameters:
        y_values (list of float): List of y-coordinates at which to slice the heatmap.
        heatmap (numpy array): The heatmap array.
        xx, yy (numpy array): Grid arrays for x and y coordinates.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    for y_value in y_values:
        # Find the closest index to the provided y_value
        y_index = np.argmin(np.abs(yy[:, 0] - y_value))

        # Extract the intensity slice along the closest y-axis row
        intensity_slice = heatmap[y_index, :]
        x_values = xx[y_index, :]

        # Plot the sliced graph in 3D
        ax.plot(x_values, [y_value] * len(x_values), intensity_slice, label=f"y={y_value:.2f}")

    # Set labels and title
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("Intensity")
    ax.set_title("3D Sliced Graph of Intensity vs. x-axis at Multiple y-values")
    ax.legend(loc='upper left')
    plt.show()
