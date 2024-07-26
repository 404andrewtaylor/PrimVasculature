import math
import heapq
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import time

# Calculate the Euclidean distance between two points in 3D space
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)

# Add edges to the priority queue based on the current point and tree label
def add_edges_to_pq(points, in_tree, pq, current_point, tree_label):
    num_points = len(points)
    for next_point in range(num_points):
        if not in_tree[next_point]:
            distance = euclidean_distance(points[current_point], points[next_point])
            heapq.heappush(pq, (distance, current_point, next_point, tree_label))

# Calculate the shortest distance between two line segments in 3D space
def distance_between_segments(p1, p2, q1, q2):
    # Vector calculations
    u = np.array(p2) - np.array(p1)
    v = np.array(q2) - np.array(q1)
    w = np.array(p1) - np.array(q1)

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a*c - b*b
    sc, tc = 0.0, 0.0

    if D < 1e-7:
        sc = 0.0
        tc = (b > c) * (d/b) + (b <= c) * (e/c)
    else:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D

    sc = max(0.0, min(1.0, sc))
    tc = max(0.0, min(1.0, tc))

    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

# Calculate the minimum spanning tree for intertwined points
def prim_mst_intertwined(points_A, points_B, edge_radius):
    num_points_A = len(points_A)
    num_points_B = len(points_B)

    in_tree_A = [False] * num_points_A
    in_tree_B = [False] * num_points_B
    mst_edges_A = []
    mst_edges_B = []
    total_weight_A = 0
    total_weight_B = 0

    pq_A = [(0, 0, 0, 'A')]  # Start from the first point of A
    pq_B = [(0, 0, 0, 'B')]  # Start from the first point of B

# Function to check if the new edge overlaps with any existing edge
    def edges_do_not_overlap(new_edge, existing_edges):
        # Iterate through each existing edge
        for edge in existing_edges:
            from_point, to_point, _ = edge
            # Calculate the distance between the two segments
            if distance_between_segments(points_A[new_edge[1]], points_A[new_edge[2]],
                                         points_B[from_point], points_B[to_point]) < 2 * edge_radius:
                return False  # Return False if there is an overlap
        return True  # Return True if no overlap is found

    # Continue until both priority queues are empty
    while pq_A or pq_B:
        # Dequeue the edge with the smallest weight from queue A
        if pq_A:
            weight_A, from_A, to_A, _ = heapq.heappop(pq_A)
            # If the edge's destination is not already in the tree and
            # there is no point in tree B within the edge radius,
            # and the new edge does not overlap with any existing edge in tree B,
            # then add it to the tree and update the total weight,
            # and add its neighbors to the priority queue
            if not in_tree_A[to_A] and not any(euclidean_distance(points_A[to_A], p) < edge_radius for p in points_B if in_tree_B[points_B.index(p)]):
                if edges_do_not_overlap((weight_A, from_A, to_A), mst_edges_B):
                    in_tree_A[to_A] = True
                    total_weight_A += weight_A
                    if from_A != to_A:
                        mst_edges_A.append((from_A, to_A, weight_A))
                    add_edges_to_pq(points_A, in_tree_A, pq_A, to_A, 'A')

        # Dequeue the edge with the smallest weight from queue B
        if pq_B:
            weight_B, from_B, to_B, _ = heapq.heappop(pq_B)
            # If the edge's destination is not already in the tree and
            # there is no point in tree A within the edge radius,
            # and the new edge does not overlap with any existing edge in tree A,
            # then add it to the tree and update the total weight,
            # and add its neighbors to the priority queue
            if not in_tree_B[to_B] and not any(euclidean_distance(points_B[to_B], p) < edge_radius for p in points_A if in_tree_A[points_A.index(p)]):
                if edges_do_not_overlap((weight_B, from_B, to_B), mst_edges_A):
                    in_tree_B[to_B] = True
                    total_weight_B += weight_B
                    if from_B != to_B:
                        mst_edges_B.append((from_B, to_B, weight_B))
                    add_edges_to_pq(points_B, in_tree_B, pq_B, to_B, 'B')

    # Return the total weights and edges of the minimum spanning trees
    return total_weight_A, mst_edges_A, total_weight_B, mst_edges_B

# Render the intertwined minimum spanning trees
def render_mst_intertwined(points_A, mst_edges_A, points_B, mst_edges_B):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points for tree A
    xs_A, ys_A, zs_A = zip(*points_A)
    ax.scatter(xs_A, ys_A, zs_A, c='r', marker='o')

    # Plot the edges for tree A
    for edge in mst_edges_A:
        from_point, to_point, _ = edge
        x_coords = [points_A[from_point][0], points_A[to_point][0]]
        y_coords = [points_A[from_point][1], points_A[to_point][1]]
        z_coords = [points_A[from_point][2], points_A[to_point][2]]
        ax.plot(x_coords, y_coords, z_coords, c='r')

    # Plot the points for tree B
    xs_B, ys_B, zs_B = zip(*points_B)
    ax.scatter(xs_B, ys_B, zs_B, c='b', marker='^')

    # Plot the edges for tree B
    for edge in mst_edges_B:
        from_point, to_point, _ = edge
        x_coords = [points_B[from_point][0], points_B[to_point][0]]
        y_coords = [points_B[from_point][1], points_B[to_point][1]]
        z_coords = [points_B[from_point][2], points_B[to_point][2]]
        ax.plot(x_coords, y_coords, z_coords, c='b')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

# Generate random points in 3D space
def generate_random_points(x_dim, y_dim, z_dim, num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, x_dim)
        y = random.uniform(0, y_dim)
        z = random.uniform(0, z_dim)
        points.append((x, y, z))
    return points

# Generate grid points based on dimensions and grid spacing
def generate_grid_points(x_dim, y_dim, z_dim, grid_spacing):
    points = []
    for x in range(0, x_dim, grid_spacing):
        for y in range(0, y_dim, grid_spacing):
            for z in range(0, z_dim, grid_spacing):
                points.append((x, y, z))
    return points

# Function to generate points within each sub-cube
def generate_points_in_subcubes(x_dim, y_dim, z_dim, subcube_dim, offset):
    points_A = []
    points_B = []
    for x in range(0, x_dim, subcube_dim):
        for y in range(0, y_dim, subcube_dim):
            for z in range(0, z_dim, subcube_dim):
                # Place a point in the center of each sub-cube for network A
                point_A = (x + subcube_dim/2, y + subcube_dim/2, z + subcube_dim/2)
                points_A.append(point_A)
                # Place a slightly offset point for network B
                point_B = (x + subcube_dim/2 + offset, y + subcube_dim/2 + offset, z + subcube_dim/2 + offset)
                points_B.append(point_B)
    return points_A, points_B

def build_adjacency_list(num_points, edges):
    adj_list = {i: [] for i in range(num_points)}
    for from_point, to_point, _ in edges:
        adj_list[from_point].append(to_point)
    return adj_list

def count_descendants_recursive(adj_list, node, descendants_count):
    if descendants_count[node] != -1:
        return descendants_count[node]
    
    count = 0
    for child in adj_list[node]:
        count += 1 + count_descendants_recursive(adj_list, child, descendants_count)
    
    descendants_count[node] = count
    return count

def calculate_all_descendants(num_points, edges):
    adj_list = build_adjacency_list(num_points, edges)
    descendants_count = [-1] * num_points  # Initialize all with -1 to indicate uncalculated
    for node in range(num_points):
        if descendants_count[node] == -1:  # If not calculated yet
            count_descendants_recursive(adj_list, node, descendants_count)
    return descendants_count

def calculate_edge_length(points, from_idx, to_idx):
    from_coords = points[from_idx]
    to_coords = points[to_idx]
    return math.sqrt((from_coords[0] - to_coords[0]) ** 2 +
                     (from_coords[1] - to_coords[1]) ** 2 +
                     (from_coords[2] - to_coords[2]) ** 2)

def calculate_width(num_descendants, length):
    return (num_descendants * length) ** (1/4)

def save_mst_to_csv(filename, points, edges, tree_label):
    descendants_count = calculate_all_descendants(len(points), edges)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['From Index', 'From X', 'From Y', 'From Z', 'To Index', 'To X', 'To Y', 'To Z', 'Weight', 'Tree', 'Descendants', 'Width'])
        for edge in edges:
            from_point_idx, to_point_idx, weight = edge
            from_coords = points[from_point_idx]
            to_coords = points[to_point_idx]
            num_descendants = descendants_count[from_point_idx]
            length = calculate_edge_length(points, from_point_idx, to_point_idx)
            width = calculate_width(num_descendants, length)
            writer.writerow([from_point_idx, *from_coords, to_point_idx, *to_coords, weight, tree_label, num_descendants, width])

def prim_mst_subcube(points_A, points_B, edge_radius):
    total_weight_A, mst_edges_A, total_weight_B, mst_edges_B = 0, [], 0, []

    num_points_A = len(points_A)
    num_points_B = len(points_B)

    in_tree_A = [False] * num_points_A
    in_tree_B = [False] * num_points_B

    pq_A = [(0, 0, 0, 'A')]  # Start from the first point of A
    pq_B = [(0, 0, 0, 'B')]  # Start from the first point of B

    while pq_A or pq_B:
        if pq_A:
            weight_A, from_A, to_A, _ = heapq.heappop(pq_A)
            if not in_tree_A[to_A] and not any(euclidean_distance(points_A[to_A], p) < edge_radius for p in points_B if in_tree_B[points_B.index(p)]):
                in_tree_A[to_A] = True
                total_weight_A += weight_A
                if from_A != to_A:
                    mst_edges_A.append((from_A, to_A, weight_A))
                add_edges_to_pq(points_A, in_tree_A, pq_A, to_A, 'A')
        
        if pq_B:
            weight_B, from_B, to_B, _ = heapq.heappop(pq_B)
            if not in_tree_B[to_B] and not any(euclidean_distance(points_B[to_B], p) < edge_radius for p in points_A if in_tree_A[points_A.index(p)]):
                in_tree_B[to_B] = True
                total_weight_B += weight_B
                if from_B != to_B:
                    mst_edges_B.append((from_B, to_B, weight_B))
                add_edges_to_pq(points_B, in_tree_B, pq_B, to_B, 'B')

    return total_weight_A, mst_edges_A, total_weight_B, mst_edges_B

# start a timer
start_time = time.time()

# Example usage with grid spacing and offset for points B
grid_spacing = 40  # Specify the grid spacing
offset = 20  # Offset for points B from points A

x_dim, y_dim, z_dim = 600, 600, 600  # Dimensions of the area
num_points = 200  # Number of random points for each tree
points_A = generate_random_points(x_dim, y_dim, z_dim, num_points)
points_B = generate_random_points(x_dim, y_dim, z_dim, num_points)
edge_radius = 1  # Radius of the edges

points_A[0] = (-10,-10,-10)
points_B[0] = (-5,-5,-5)

#points_A = generate_grid_points(x_dim, y_dim, z_dim, grid_spacing)

# Offset points B from points A
#points_B = [(x + offset, y + offset, z + offset) for x, y, z in points_A]

total_weight_A, mst_edges_A, total_weight_B, mst_edges_B = prim_mst_intertwined(points_A, points_B, edge_radius)

print("Total weight of MST for tree A:", total_weight_A)
print("Edges in MST for tree A:")
for edge in mst_edges_A:
    from_point, to_point, weight = edge
    print(f"From {from_point} to {to_point} with weight {weight}")

print("Total weight of MST for tree B:", total_weight_B)
print("Edges in MST for tree B:")
for edge in mst_edges_B:
    from_point, to_point, weight = edge
    print(f"From {from_point} to {to_point} with weight {weight}")

# Save MST edges to CSV files
save_mst_to_csv('mst_edges_A.csv', points_A, mst_edges_A, 'A')
save_mst_to_csv('mst_edges_B.csv', points_B, mst_edges_B, 'B')

# Render the intertwined minimum spanning trees
#render_mst_intertwined(points_A, mst_edges_A, points_B, mst_edges_B)

# end the timer
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")



