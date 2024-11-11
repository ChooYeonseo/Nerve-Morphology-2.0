class Node:
    def __init__(self, name, parent = [], children = []):
        self.value = name
        self.parent = parent
        self.children = children

        assert all(isinstance(i, Node) for i in self.parent), "Not all elements in parent list are Node."
        assert all(isinstance(j, Node) for j in self.children), "Not all elements in parent list are Node."

    def add_child(self, child_node):
        assert isinstance(child_node, Node)
        self.children.append(child_node)
        child_node.parent.append(self)

    def isRoot(self):
        if len(self.parent) == 0 :
            return True
        
    def set_coord(self, P):
        x, y = P[0], P[1]
        self.coord = [x, y]

    def __getattribute__(self, name: str):
        print(f"Accessing attribute: {name}")
        return super().__getattribute__(name)
