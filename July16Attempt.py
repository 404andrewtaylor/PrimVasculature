import math
import heapq
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)

def prim_mst(points):
    num_points = len(points)
    if num_points == 0:
        return 0, []
    
    # Initialize
    in_tree = [False] * num_points
    mst_edges = []
    total_weight = 0

    # Priority queue to store (weight, from_point, to_point)
    pq = [(0, 0, 0)]  # Start from the first point

    while pq:
        weight, from_point, to_point = heapq.heappop(pq)
        if in_tree[to_point]:
            continue

        in_tree[to_point] = True
        total_weight += weight
        if from_point != to_point:
            mst_edges.append((from_point, to_point, weight))

        for next_point in range(num_points):
            if not in_tree[next_point]:
                distance = euclidean_distance(points[to_point], points[next_point])
                heapq.heappush(pq, (distance, to_point, next_point))

    return total_weight, mst_edges

def render_mst(points, mst_edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # Plot the edges
    for edge in mst_edges:
        from_point, to_point, _ = edge
        x_coords = [points[from_point][0], points[to_point][0]]
        y_coords = [points[from_point][1], points[to_point][1]]
        z_coords = [points[from_point][2], points[to_point][2]]
        ax.plot(x_coords, y_coords, z_coords, c='b')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

def generate_grid_points(x_dim, y_dim, z_dim):
    points = []
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                points.append((x, y, z))
    return points

def generate_random_points(x_dim, y_dim, z_dim, num_points):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, x_dim)
        y = random.uniform(0, y_dim)
        z = random.uniform(0, z_dim)
        points.append((x, y, z))
    return points

# Example usage
x_dim, y_dim, z_dim = 60, 60, 60  # Dimensions of the area
num_points = 10000  # Number of random points to generate
points = generate_random_points(x_dim, y_dim, z_dim, num_points)

total_weight, mst_edges = prim_mst(points)

print("Total weight of MST:", total_weight)
print("Edges in MST:")
for edge in mst_edges:
    from_point, to_point, weight = edge
    print(f"From {from_point} to {to_point} with weight {weight}")

# Render the MST
render_mst(points, mst_edges)
