import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
import random
import math

def generate_3d_grid_points(x, y, z):
    points = [(i, j, k) for i in range(x) for j in range(y) for k in range(z)]
    return points

def construct_tree_with_constraints(G, start_point, points, max_connections):
    G.add_node(start_point)
    available_points = set(points)
    connected = set()
    queue = [start_point]

    while queue and available_points:
        current_point = queue.pop(0)

        if len(list(G.neighbors(current_point))) < max_connections:
            possible_connections = list(available_points - set(G.neighbors(current_point)))
            random.shuffle(possible_connections)
            for point in possible_connections[:max_connections - len(list(G.neighbors(current_point)))]:
                if not will_cross_edges(G, current_point, point):
                    G.add_edge(current_point, point)
                    connected.add(point)
                    queue.append(point)
                    available_points.remove(point)

    return connected

def will_cross_edges(G, p1, p2):
    for edge in G.edges:
        q1, q2 = edge
        if lines_intersect(p1, p2, q1, q2):
            return True
    return False

def lines_intersect(p1, p2, q1, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]) and
            q[2] <= max(p[2], r[2]) and q[2] >= min(p[2], r[2])):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    if o2 == 0 and on_segment(p1, q2, p2):
        return True

    if o3 == 0 and on_segment(q1, p1, q2):
        return True

    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False

def find_shortest_path_3d_grid(x, y, z, start_point):
    points = generate_3d_grid_points(x, y, z)
    G = nx.Graph()

    # Construct the tree starting from the start_point
    construct_tree_with_constraints(G, start_point, points, 3)

    # Find a path covering all nodes
    path = list(nx.dfs_preorder_nodes(G, source=start_point))

    # Optimize the path using simulated annealing
    def path_length(path):
        total_length = 0
        for i in range(len(path) - 1):
            total_length += nx.shortest_path_length(G, path[i], path[i + 1])
        return total_length

    def swap(path, i, j):
        new_path = path[:]
        new_path[i], new_path[j] = new_path[j], new_path[i]
        return new_path

    current_path = path
    current_length = path_length(current_path)

    # Simulated annealing parameters
    initial_temperature = 100.0
    cooling_rate = 0.99
    iterations = 1000

    # Simulated annealing loop
    for iteration in range(iterations):
        temperature = initial_temperature * math.exp(-cooling_rate * iteration)
        if len(path) > 2:
            i, j = random.sample(range(1, len(path)), 2)  # Ensure i < j
        else:
            i, j = 0, 1

        new_path = swap(current_path, i, j)
        new_length = path_length(new_path)
        
        # Calculate acceptance probability
        if new_length < current_length or (temperature > 0 and random.random() < math.exp(-(new_length - current_length) / temperature)):
            current_path = new_path
            current_length = new_length

    shortest_path = [(current_path[i], current_path[i + 1]) for i in range(len(current_path) - 1)]

    return shortest_path

def plot_3d_grid(points, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    for point in points:
        ax.scatter(*point, color='b')

    # Plot edges without crossings
    plotted_edges = set()
    for edge in edges:
        p1, p2 = edge
        if (p1, p2) not in plotted_edges and (p2, p1) not in plotted_edges:
            ax.add_line(Line3D((p1[0], p2[0]), (p1[1], p2[1]), (p1[2], p2[2]), color='r'))
            plotted_edges.add((p1, p2))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

# Example usage:
if __name__ == "__main__":
    # Input dimensions for x, y, z
    x = int(input("Enter number of points along X dimension: "))
    y = int(input("Enter number of points along Y dimension: "))
    z = int(input("Enter number of points along Z dimension: "))

    # Specify the start point
    start_point = (0, 0, 0)  # Example: Start point at the origin (0, 0, 0)

    # Find shortest path
    shortest_path = find_shortest_path_3d_grid(x, y, z, start_point)
    if shortest_path:
        print("Shortest path edges:", shortest_path)
        # Plot the 3D grid and shortest path
        plot_3d_grid(generate_3d_grid_points(x, y, z), shortest_path)
