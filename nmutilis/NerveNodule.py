import numpy as np
import math
from nmutilis.bone_model import Bone

class Node:
    def __init__(self, name, children = [], coord = None):
        self.value = name
        self.children = children if children else []
        self.coord = coord  # Coordinate tuple (x, y)

        assert all(isinstance(j, Node) for j in self.children), "Not all elements in parent list are Node."

    def add_child(self, child_node):
        assert isinstance(child_node, Node), "The child_node must be an instance of Node."
        if child_node not in self.children:  # Checks for duplicates properly
            self.children.append(child_node)

    def get_children_ids(self):
        Nl = []
        for i in self.children:
            Nl.append(i.get_name())
        return Nl

    def get_name(self):
        return self.value

    def get_coord(self):
        return self.coord

    # def __getattribute__(self, name: str):
    #     """Override to track attribute access"""
    #     print(f"Accessing attribute: {name}")
    #     return super().__getattribute__(name)

class Nerve():
    def __init__(self, code, raw_data:dict, helper_data:dict, bone:Bone):
        self.name = code
        self.nodes = {}
        self.raw_data = raw_data
        self.helper_data = helper_data
        self.bone = bone

        self.landmark_list = ['mid10_sn', 'mid12_sn', 'mid_sn', 'tip_sn', 'tip_cp_recal', 'brch1', 'brchA2', 'brchP2', 'sp']

        imp_data = {}
        for i in self.raw_data:
            for j in self.landmark_list:
                if (j in i) and ('tk' not in i) and ('brch' not in i) and (isinstance(self.raw_data[i], float))and not math.isnan(self.raw_data[i]):
                    imp_data[i] = self.raw_data[i]
        
        for i in imp_data:
            imp_data[i] = self.rel(imp_data[i])
            if self.landmark_list[0] in i:
                self.nodes[i] = Node(i, coord=(self.bone.get_mid10()[1][0] + imp_data[i], self.bone.get_mid10()[1][1]))
            if self.landmark_list[1] in i:
                self.nodes[i] = Node(i, coord=(self.bone.get_mid12()[1][0] + imp_data[i], self.bone.get_mid12()[1][1]))
            if self.landmark_list[2] in i:
                self.nodes[i] = Node(i, coord=(-imp_data[i], self.bone.get_MID()[1]))
            if self.landmark_list[3] in i:
                self.nodes[i] = Node(i, coord=(self.bone.get_TIP()[0] - imp_data[i], self.bone.get_TIP()[1]))
        
        temp = {}
     
        for i in self.helper_data:
            for j in self.landmark_list:
                if (j in i) and (isinstance(self.helper_data[i], float))and not math.isnan(self.helper_data[i]):
                    temp[i] = self.helper_data[i]
        
        for i in temp:
            if self.landmark_list[4] in i:
                self.nodes['tip_cp'] = Node('tip_cp', coord=((temp[i]+148)/38 * self.bone.get_scale(),temp[i] * self.bone.get_scale()))
            if self.landmark_list[5] in i:
                name = self.landmark_list[5]
                self.nodes[name] = Node(name, coord=(temp['brch1_x'] * self.bone.get_scale(), temp['brch1_y'] * self.bone.get_scale()))
            if self.landmark_list[6] in i:
                name = self.landmark_list[6]
                self.nodes[name] = Node(name, coord=(temp['brchA2_x'] * self.bone.get_scale(), temp['brchA2_y'] * self.bone.get_scale()))
            if self.landmark_list[7] in i:
                name = self.landmark_list[7]
                self.nodes[name] = Node(name, coord=(temp['brchP2_x'] * self.bone.get_scale(), temp['brchP2_y'] * self.bone.get_scale()))
            if self.landmark_list[8] in i:
                name = self.landmark_list[8]
                self.nodes[name] = Node(name, coord=(temp['sp_x'] * self.bone.get_scale(), temp['sp_y'] * self.bone.get_scale()))
        
        print('Total {} nodes have been located. Please refer to the below and relate each nodes.'.format(len(self.nodes)))
        for i in self.nodes:
            print('{}\'s coordinate: {}'.format(i, self.nodes[i].get_coord()))

    def add_relation(self, parent_id, child_id):
        parent_node = self.nodes[parent_id]
        child_node = self.nodes[child_id]
        parent_node.add_child(child_node)

    def get_name(self):
        return self.name

        
    def get_nodes(self):
        return self.nodes

    def rel(self, value):
        w = self.raw_data['width_mm']
        return self.bone.get_width() * value / (w)

    def get_branch_lines(self):
        """Generate line segments for each branch as coordinate arrays."""
        branches = []
        # print(self.nodes.values())
        for node in self.nodes.values():
            if node.get_coord() and node.children:
                for child in node.children:
                    if child.get_coord():
                        branches.append(np.linspace(node.get_coord(), child.get_coord()))
        return branches
    
    def save_branch(self, file_dir):
        branches = self.get_branch_lines()
        np.savez(file_dir, *branches)

