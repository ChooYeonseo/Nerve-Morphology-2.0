import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree  # Import KDTree for faster nearest neighbor search
from tqdm import tqdm
import pprint as pprint


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

def plot_heatmap(lims, bone, sigma, branch_list, name_list):
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


    # Plotting the heatmap
    plt.figure(figsize=(5, 8))
    plt.gca().set_aspect('equal')

    # Iterate over the lines
    heatmap = np.zeros_like(xx)
    contribution_tracker = np.zeros_like(xx)


    for i in tqdm(name_list):
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
    plt.xticks([nxlim, pxlim], ['Anterior', 'Posterior'])
    plt.yticks([nylim, pylim], ['Inferior', 'Superior'])


    plt.title("Probabilistic Heatmap of Nerve with Branches")
    plt.legend()
    plt.grid(True)

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


    plt.show()

    return heatmap, xx, yy

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
        print(f"The {n}th maximum intensity is {nth_max_value} at ({nth_max_x}, {nth_max_y})")

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
        plt.scatter(nth_max_x, nth_max_y, color="blue", s=50, zorder=5, label=f"{n}th Max Point")
        plt.annotate(f"{n}th Max ({nth_max_x:.2f}, {nth_max_y:.2f})",
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
    plt.scatter(nth_max_x, nth_max_y, color="blue", s=50, zorder=5, label=f"{n}th Max Point")
    plt.annotate(f"{n}th Max ({nth_max_x:.2f}, {nth_max_y:.2f})",
                 (nth_max_x, nth_max_y),
                 textcoords="offset points", xytext=(-20, 10),
                 color="blue", fontsize=10, arrowprops=dict(arrowstyle="->", color="blue"))

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(f"Magnified View of Specified Area with {n}th Max Point and Bone Section")
    plt.legend()
    plt.grid(True)
    plt.show()

    return nth_max_x, nth_max_y, nth_max_value
