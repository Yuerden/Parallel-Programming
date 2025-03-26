import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Constants
MAX_CORE_DOUBLINGS = 20
MAX_NODES = 1 << (MAX_CORE_DOUBLINGS // 2)  # 2^10
MAX_CORES_PER_NODE = 1 << (MAX_CORE_DOUBLINGS // 2)  # 2^10
MAX_F = 5
f_values = [0.99999999, 0.9999, 0.99, 0.9, 0.5]

def perf(r):
    return np.sqrt(r)

def compute_speedup(f_index):
    f = f_values[f_index]
    x, y, z_asymmetric, z_dynamic = [], [], [], []

    for node_index in range(1, MAX_NODES + 1, 4):
        for r_index in range(1, MAX_CORES_PER_NODE + 1, 4):
            for core_index in range(r_index, MAX_CORES_PER_NODE + 1, 4):
                total_cores = node_index * core_index
                speedup_asymmetric = node_index / ((1.0 - f) / perf(r_index) + f / (perf(r_index) + core_index - r_index))
                speedup_dynamic = node_index / ((1.0 - f) / perf(r_index) + f / core_index)
                
                x.append(total_cores)
                y.append(r_index)
                z_asymmetric.append(speedup_asymmetric)
                z_dynamic.append(speedup_dynamic)
                
    return x, y, z_asymmetric, z_dynamic

# Plotting
for i in range(MAX_F):
    x, y, z_asymmetric, z_dynamic = compute_speedup(i)

    fig = plt.figure(figsize=(12, 6))

    # Asymmetric plot
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x, y, z_asymmetric, color='b')
    ax.set_title(f'Speedup Asymmetric for f = {f_values[i]}')
    ax.set_xlabel('Total Cores (N * c)')
    ax.set_ylabel('R Cores')
    ax.set_zlabel('Speedup Asymmetric')

    # Dynamic plot
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x, y, z_dynamic, color='r')
    ax.set_title(f'Speedup Dynamic for f = {f_values[i]}')
    ax.set_xlabel('Total Cores (N * c)')
    ax.set_ylabel('R Cores')
    ax.set_zlabel('Speedup Dynamic')

    plt.show()
