import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# Параметры
map_width = 10
map_height = 5
start = np.array([0, 0])
end = np.array([9, 4])
num_samples = 1000

# Препятствия
circle_center = np.array([3, 3])
circle_radius = 1.5
theta = np.linspace(0, 2*np.pi, 100)
circle_x = circle_radius * np.cos(theta) + circle_center[0]
circle_y = circle_radius * np.sin(theta) + circle_center[1]
polygon_x = [6, 8, 8, 5, 5, 7, 7, 6, 6]
polygon_y = [1, 1, 4, 4, 3, 3, 2, 2, 1]
polygon = Path(np.vstack((polygon_x, polygon_y)).T)

# Случайные точки и Вороной
np.random.seed(0)
xn = np.random.rand(num_samples) * map_width
yn = np.random.rand(num_samples) * map_height
points = np.vstack((xn, yn)).T
vor = Voronoi(points)

# === График 1: Вороной до фильтрации ===
vx_raw, vy_raw = [], []
for vpair in vor.ridge_vertices:
    if -1 in vpair:
        continue
    v0, v1 = vor.vertices[vpair]
    if np.any(v0 < 0) or np.any(v0 > [map_width, map_height]) or np.any(v1 < 0) or np.any(v1 > [map_width, map_height]):
        continue
    vx_raw.append([v0[0], v1[0]])
    vy_raw.append([v0[1], v1[1]])

fig, ax = plt.subplots()
ax.plot(xn, yn, 'k.')
ax.plot(circle_x, circle_y, 'r-')
ax.plot(polygon_x, polygon_y, 'r-')
ax.plot(start[0], start[1], 'bx', markersize=10)
ax.plot(end[0], end[1], 'ro', markersize=10)
for xseg, yseg in zip(vx_raw, vy_raw):
    ax.plot(xseg, yseg, 'b-')
ax.set_aspect('equal')
ax.set_title('Вороной до фильтрации')
plt.show()

# === Фильтрация рёбер ===
vx_clean, vy_clean = [], []
for vpair in vor.ridge_vertices:
    if -1 in vpair:
        continue
    v0, v1 = vor.vertices[vpair]
    if np.any(v0 < 0) or np.any(v0 > [map_width, map_height]) or np.any(v1 < 0) or np.any(v1 > [map_width, map_height]):
        continue
    mid = (v0 + v1) / 2
    if np.linalg.norm(mid - circle_center) < circle_radius:
        continue
    if polygon.contains_point(mid):
        continue
    vx_clean.append([v0[0], v1[0]])
    vy_clean.append([v0[1], v1[1]])

# === График 2: Вороной после фильтрации ===
fig, ax = plt.subplots()
ax.plot(xn, yn, 'k.')
ax.plot(circle_x, circle_y, 'r-')
ax.plot(polygon_x, polygon_y, 'r-')
ax.plot(start[0], start[1], 'bx', markersize=10)
ax.plot(end[0], end[1], 'ro', markersize=10)
for xseg, yseg in zip(vx_clean, vy_clean):
    ax.plot(xseg, yseg, 'b-')
ax.set_aspect('equal')
ax.set_title('Вороной после фильтрации')
plt.show()

# === Построение графа ===
xy_1 = np.array(vx_clean).T
xy_2 = np.array(vy_clean).T
xy_12 = np.vstack((xy_1.T, xy_2.T))
node_coords, _, node_indices = np.unique(xy_12, axis=0, return_index=True, return_inverse=True)

num_edges = len(vx_clean)
st_node_index = node_indices[:num_edges]
ed_node_index = node_indices[num_edges:]
edge_lengths = np.linalg.norm(xy_1 - xy_2, axis=0)

row = np.concatenate([st_node_index, ed_node_index])
col = np.concatenate([ed_node_index, st_node_index])
data = np.concatenate([edge_lengths, edge_lengths])
graph = csr_matrix((data, (row, col)), shape=(len(node_coords), len(node_coords)))

def closest_node(pt, nodes):
    return np.argmin(np.sum((nodes - pt)**2, axis=1))

start_node = closest_node(start, node_coords)
end_node = closest_node(end, node_coords)

distances, predecessors = dijkstra(csgraph=graph, indices=start_node, return_predecessors=True)
path = []
i = end_node
while i != start_node:
    path.append(i)
    i = predecessors[i]
    if i == -9999:
        path = []
        break
if path:
    path.append(start_node)
    path = path[::-1]
    opt_path = node_coords[path]
else:
    opt_path = None

# === График 3: Кратчайший путь ===
fig, ax = plt.subplots()
ax.plot(circle_x, circle_y, 'r-')
ax.plot(polygon_x, polygon_y, 'r-')
ax.plot(start[0], start[1], 'bx', markersize=10)
ax.plot(end[0], end[1], 'ro', markersize=10)
for xseg, yseg in zip(vx_clean, vy_clean):
    ax.plot(xseg, yseg, 'b-', alpha=0.3)
if opt_path is not None:
    ax.plot(opt_path[:, 0], opt_path[:, 1], 'g-', linewidth=2)
ax.set_aspect('equal')
ax.set_title('Кратчайший путь через граф Вороного')
plt.show()