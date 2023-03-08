"""Path planning and visualisation"""

from random import randint
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

class GraphXY:
    """Simple 2D maps for visualising path planning algorithms"""

    def __init__(self, nrows, ncols, obstacle_coords=None, origin=None, target=None):
        """Initialise map object of size (nrows, ncols)"""
        self.nrows = nrows
        self.ncols = ncols
        self.shape = (nrows, ncols)
        self.size = nrows * ncols
        self.nodes = np.zeros(self.shape)
        
        if obstacle_coords is None:
            # randomly generate obstacles in the map if no
            # obstacle coordinates are provided
            fraction_blocked = 0.4 # ~40% of map populated with obstacles
            self.obstacles = self.generate_obstacles(fraction_blocked)
        else:
            # set user input obstacles
            self.obstacles = obstacles
            for coord in obstacle_coords:
                self.nodes[coord] = np.inf
        
        if origin is None:
            self.origin = self.select_random_coord()
        
        if target is None:
            self.target = self.select_random_coord()
        
        self.set_origin(self.origin)
        self.set_target(self.target)

    def __repr__(self):
        """Display map"""
        return str(self.nodes)

    def select_random_coord(self):
        """Returns the coordinates of a random (open) node in the map"""
        if 0 not in self.nodes:
            raise ValueError("No open nodes (i.e. not obstacle/origin/target)")
        
        while True:
            i = randint(0, self.nrows-1)
            j = randint(0, self.ncols-1)
            coord = (i, j)
            if self.nodes[coord] == 0:
                return coord

    def generate_obstacles(self, fraction_blocked):
        """Generate some quaisi-random obstacles in the Map"""
        num_blocked = int(round(self.size * fraction_blocked))
        obstacles = []
        for _ in range(num_blocked):
            coord = self.select_random_coord()
            obstacles.append(coord)
            self.nodes[coord] = np.inf
        return obstacles

    def set_origin(self, coord):
        """Set source node"""
        self.nodes[coord] = 2

    def set_target(self, coord):
        """Set target node"""
        self.nodes[coord] = 3

    def all_coords(self):
        """Returns a generator which yields all the coordinates in the map"""
        coord_generator = product(
            range(self.nrows),
            range(self.ncols)
        )
        return coord_generator

    def out_of_bounds(self, coord):
        """Determines whether a coordinate is out of bounds"""
        out_of_bounds = False
        if coord[0] < 0 or coord[0] >= self.nrows:
            out_of_bounds = True
            
        elif coord[1] < 0 or coord[1] >= self.ncols:
            out_of_bounds = True
            
        elif coord in self.obstacles:
            out_of_bounds = True
            
        return out_of_bounds
    
    def mark_path(self, path):
        """Mark nodes on path in self.nodes for plotting"""
        path_coordinates = tuple(zip(*path))
        self.nodes[path_coordinates] = 4
        
    def plot(self):
        """Plot graph using matplotlib"""
        return plot_map(self.nodes, cmap="viridis")
        

def manhattan_dist(point1, point2):
    """Find the Manhattan distance between two points in 2d cartesian coords"""
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2, ord=1)


def euclidean_dist(point1, point2):
    """Find the Euclidean distance between two points in 2d cartesian coords"""
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1-point2, ord=2)


def neighbouring_nodes(point, map_obj):
    """Returns a list of neighbouring nodes to the point passed"""
    neighbour_generator = product(
        range(point[0]-1, point[0]+2),
        range(point[1]-1, point[1]+2)
    )
    neighbours = [coord for coord in neighbour_generator if not map_obj.out_of_bounds(coord)]
    return neighbours


def update_distance(dist, current_node, next_node):
    """Calculate and compare new and old distances to a node"""
    new_dist = dist[current_node] + euclidean_dist(current_node, next_node)
    if new_dist < dist[next_node]:
        dist[next_node] = new_dist
    return dist
    

def dijkstra(map_obj):
    """Dijkstra's algorithm"""

    # set up dist array
    dist = np.full(map_obj.shape, np.inf)
    dist[map_obj.origin] = 0
    
    # Set up prioritised queue
    queue = [coord for coord in map_obj.all_coords() if coord not in map_obj.obstacles]
    queue.sort(key=lambda coord: euclidean_dist(map_obj.origin, coord), reverse=True)   

    while queue:
        node = queue.pop()
        if node == map_obj.target:
            break
        
        neighbours = neighbouring_nodes(node, map_obj)
        neighbours = [nbr for nbr in neighbours if nbr in queue]

        for neighbour in neighbours:
            dist = update_distance(dist, node, neighbour)
        
        queue.sort(key=lambda coord: dist[coord], reverse=True)
        
        yield dist # yield facilitates animation
    # return dist


def get_path(map_obj, dist):
    """Returns the shortest path to the target"""
    # TODO fix infinite loop if target is unreachable
    current_node = map_obj.target
    path = []
    
    while current_node != map_obj.origin:
        neighbours = neighbouring_nodes(current_node, map_obj)      
        neighbours.sort(key=lambda coord: dist[coord], reverse=True)
        next_step = neighbours.pop()
        if next_step != map_obj.origin:
            path.append(next_step)
        current_node = next_step
        
    return path


def plot_map(map_array, **kwargs):
    """Generate a visualisation of the map using matplotlib"""
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    ax.set_aspect("equal")
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.spines[:].set_color("red")
    ax.tick_params(
        axis="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.pcolormesh(map_array, **kwargs)
    return (fig, ax)