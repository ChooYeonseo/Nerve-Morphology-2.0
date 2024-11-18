import numpy as np

class Node:
    def __init__(self, id, coord=None):
        self.id = id  # Unique identifier for the node
        self.coord = coord  # Coordinate tuple (x, y)
        self.parent = None  # Parent node
        self.children = []  # List of child nodes

    def set_coord(self, coord):
        self.coord = coord

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

class Nerve:
    def __init__(self, name):
        self.name = name  # Name of the nerve
        self.nodes = {}  # Dictionary to store nodes by their ID

    def add_node(self, id, coord=None):
        if id in self.nodes:
            raise ValueError(f"Node with ID {id} already exists in {self.name}.")
        node = Node(id, coord)
        self.nodes[id] = node
        return node

    def get_node(self, id):
        return self.nodes.get(id, None)

    def add_relation(self, parent_id, child_id):
        parent_node = self.get_node(parent_id)
        child_node = self.get_node(child_id)
        if not parent_node or not child_node:
            raise ValueError(f"Both nodes {parent_id} and {child_id} must exist to add a relation.")
        parent_node.add_child(child_node)

    def get_branch_lines(self):
        """Generate line segments for each branch as coordinate arrays."""
        branches = []
        for node in self.nodes.values():
            if node.coord and node.children:
                for child in node.children:
                    if child.coord:
                        branches.append(np.linspace(node.coord, child.coord, num=100))
        return branches

# Example usage
if __name__ == "__main__":
    # Create a Nerve instance for N1
    N1 = Nerve("N1")

    # Add nodes to N1
    N1.add_node(1, coord=(0, 0))
    N1.add_node(2, coord=(-1, -1))
    N1.add_node(3, coord=(1, -1))
    N1.add_node(4, coord=(-1.5, -2))
    N1.add_node(5, coord=(-0.5, -2))
    N1.add_node(6, coord=(0.5, -2))
    N1.add_node(7, coord=(1.5, -2))

    # Define relationships
    N1.add_relation(1, 2)
    N1.add_relation(1, 3)
    N1.add_relation(2, 4)
    N1.add_relation(2, 5)
    N1.add_relation(3, 6)
    N1.add_relation(3, 7)

    # Get branches as coordinate arrays
    branches = N1.get_branch_lines()

    # Print branch segments
    for i, branch in enumerate(branches):
        print(f"Branch {i+1}: {branch[0]} to {branch[-1]}")