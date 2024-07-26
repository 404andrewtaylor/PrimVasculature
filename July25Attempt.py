import math
import heapq
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
from scipy.optimize import curve_fit

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

# Generate random points in 3D space
def generate_random_points(x_dim, y_dim, z_dim, num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, x_dim)
        y = random.uniform(0, y_dim)
        z = random.uniform(0, z_dim)
        points.append((x, y, z))
    return points

# Measure runtime of the MST algorithm for different numbers of points
def estimate_runtime(x_dim, y_dim, z_dim, max_points, step, edge_radius):
    runtimes = []
    num_points_range = list(range(1, max_points + 1, step))
    for num_points in num_points_range:
        points_A = generate_random_points(x_dim, y_dim, z_dim, num_points)
        points_B = generate_random_points(x_dim, y_dim, z_dim, num_points)
        
        start_time = time.time()
        prim_mst_intertwined(points_A, points_B, edge_radius)
        end_time = time.time()
        
        runtime = end_time - start_time
        runtimes.append(runtime)
        print(f"Number of points: {num_points}, Runtime: {runtime:.4f} seconds")
        
    return num_points_range, runtimes

# Define a function to fit the runtime data
def fit_runtime_function(num_points, a, b, c):
    return a * num_points**2 + b * num_points + c

# Run the estimation and fit the data
x_dim, y_dim, z_dim = 1000, 1000, 1000
max_points = 500
step = 10
edge_radius = 1

num_points_range, runtimes = estimate_runtime(x_dim, y_dim, z_dim, max_points, step, edge_radius)

# Fit the collected runtime data to a quadratic function
params, _ = curve_fit(fit_runtime_function, num_points_range, runtimes)
a, b, c = params

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.plot(num_points_range, runtimes, 'o', label='Measured Runtimes')
plt.plot(num_points_range, fit_runtime_function(np.array(num_points_range), a, b, c), '-', label=f'Fit: {a:.2e}n² + {b:.2e}n + {c:.2e}')
plt.xlabel('Number of Points')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime Estimation of MST Algorithm')
plt.legend()
plt.grid(True)
plt.show()

print(f"Estimated function: Runtime(n) = {a:.2e} * n^2 + {b:.2e} * n + {c:.2e}")
