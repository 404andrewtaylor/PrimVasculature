import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from collections import deque

def generate_3d_grid(dimensions):
    x, y, z = dimensions
    return np.zeros((x, y, z), dtype=int)

def is_valid_point(point, dimensions):
    x, y, z = point
    max_x, max_y, max_z = dimensions
    return 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z

def get_neighbors(point, dimensions):
    x, y, z = point
    neighbors = []
    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        neighbor = (x + dx, y + dy, z + dz)
        if is_valid_point(neighbor, dimensions):
            neighbors.append(neighbor)
    return neighbors

def connect_pipes_3d(grid, start_point):
    dimensions = grid.shape
    visited = np.zeros(dimensions, dtype=bool)
    queue = deque([start_point])
    visited[start_point] = True

    while queue:
        current_point = queue.popleft()
        neighbors = get_neighbors(current_point, dimensions)

        connected_neighbors = 0
        for neighbor in neighbors:
            if not visited[neighbor] and connected_neighbors < 3:  # Limit junctions to 3 pipes
                grid[current_point] += 1
                grid[neighbor] += 1
                visited[neighbor] = True
                queue.append(neighbor)
                connected_neighbors += 1

    return grid, visited

def plot_3d_connections(grid, visited):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    edges = []
    x_dim, y_dim, z_dim = grid.shape

    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if visited[x, y, z]:
                    neighbors = get_neighbors((x, y, z), (x_dim, y_dim, z_dim))
                    for nx, ny, nz in neighbors:
                        if visited[nx, ny, nz]:
                            edges.append([(x, y, z), (nx, ny, nz)])

    edge_colors = ['b' for _ in range(len(edges))]
    edge_collection = Line3DCollection(edges, colors=edge_colors, linewidths=2)
    ax.add_collection3d(edge_collection)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Root-like Network Connections')
    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, y_dim)
    ax.set_zlim(0, z_dim)

    plt.show()

# Input dimensions of the 3D grid
x = int(input("Enter the dimension x: "))
y = int(input("Enter the dimension y: "))
z = int(input("Enter the dimension z: "))
dimensions = (x, y, z)

# Generate the 3D grid
grid = generate_3d_grid(dimensions)

# Starting point
start_point = (0, 0, 0)

# Connect pipes starting from the specified point
grid, visited = connect_pipes_3d(grid, start_point)

# Plot the 3D connections
plot_3d_connections(grid, visited)
