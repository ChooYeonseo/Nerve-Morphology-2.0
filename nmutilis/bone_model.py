import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import CubicSpline

class Bone:
    def __init__(self, scale=1, sampling_num=1000, bone_polygon_under = None, bone_polygon_upper = None):
        self.scale = scale
        self.sampling_num = sampling_num

        # Strict Bone Model. 
        # Meaning this bone model will be scaled but original shape will never change.
        # No need for change, but still we've made options for those who are willing to alter the tibia shape.
        if bone_polygon_under == None:
            self.bone_polygon_under = np.array([[1, 5], [0, 0], [1, -2], [2, -2.5], [3, -2], [4, -2], [5, 0], [4, 4]]) * self.scale
        if bone_polygon_upper == None:
            self.bone_polygon_upper = np.array([[5, 42], [6, 44], [6, 46], [1, 47], [-1, 45], [0, 42]]) * self.scale

        under_x = self.bone_polygon_under[:, 0]
        under_y = self.bone_polygon_under[:, 1]



        t = np.linspace(0, 1, len(under_x))
        t_new = np.linspace(0, 1, self.sampling_num)

        under_spline_x = CubicSpline(t, under_x, bc_type='natural') 
        under_spline_y = CubicSpline(t, under_y, bc_type='natural')
        self.under_x_smooth = under_spline_x(t_new)
        self.under_y_smooth = under_spline_y(t_new)

        # upper head
        upper_x = self.bone_polygon_upper[:, 0]
        upper_y = self.bone_polygon_upper[:, 1]

        t = np.linspace(0, 1, len(upper_x))
        t_new = np.linspace(0, 1, self.sampling_num)

        upper_spline_x = CubicSpline(t, upper_x, bc_type='natural') 
        upper_spline_y = CubicSpline(t, upper_y, bc_type='natural')
        upper_x_smooth = upper_spline_x(t_new)
        upper_y_smooth = upper_spline_y(t_new)

        # shaft
        self.medial_border_x = np.linspace(self.bone_polygon_under[-1][0], self.bone_polygon_upper[0][0], self.sampling_num)
        self.medial_border_y = np.linspace(self.bone_polygon_under[-1][1], self.bone_polygon_upper[0][1], self.sampling_num)
        anterior_border_x = np.linspace(self.bone_polygon_upper[-1][0], self.bone_polygon_under[0][0], self.sampling_num)
        anterior_border_y = np.linspace(self.bone_polygon_upper[-1][1], self.bone_polygon_under[0][1], self.sampling_num)

        self.x = np.concatenate([self.under_x_smooth, self.medial_border_x, upper_x_smooth, anterior_border_x])
        self.y = np.concatenate([self.under_y_smooth, self.medial_border_y, upper_y_smooth, anterior_border_y])

        assert self.x.shape[0] == self.sampling_num * 4
        assert self.y.shape[0] == self.sampling_num * 4

    def get_mid10(self):
        target = 5624/1445 + 8*(209011)**0.5/10115
        target *= self.scale

        closest_index = np.argmin(np.abs(self.medial_border_x - target))
        closest_index += self.under_x_smooth.shape[0]

        return closest_index, (self.x[closest_index], self.y[closest_index])
    
    def get_mid12(self):
        target = 5624/1445 + 4*(1233419)**0.5/10115
        target *= self.scale

        closest_index = np.argmin(np.abs(self.medial_border_x - target))
        closest_index += self.under_x_smooth.shape[0]

        return closest_index, (self.x[closest_index], self.y[closest_index])
    
    def get_MID(self):
        # MID will be the center point of every coordinate.
        self.mid_coord = self.bone_polygon_under[1]
        return (self.mid_coord[0], self.mid_coord[1])

    def get_TIP(self):
        self.tip_coord = self.bone_polygon_under[3]
        return (self.tip_coord[0], self.tip_coord[1])

    def get_LAT(self):
        self.lat_coord = self.bone_polygon_under[-2]
        return (self.lat_coord[0], self.lat_coord[1])
    
    def get_width(self):
        return self.get_LAT()[0] - self.get_MID()[0]
    
    def get_scale(self):
        return self.scale
    
    def plot_bone(self):
        plt.plot(self.x, self.y, label='bone', color='black')
        # Show the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()