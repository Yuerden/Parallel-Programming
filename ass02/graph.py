import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the performance results
df = pd.read_csv('performance_results.csv', names=['blocksize', 'worldsize', 'time', 'updates_per_second'])

# Set the plot style
# plt.style.use('seaborn-darkgrid')
# plt.style.use('seaborn')


# Unique block sizes and world sizes for plotting
block_sizes = df['blocksize'].unique()
world_sizes = df['worldsize'].unique()

# Plot execution time vs. block size for each world size
for world_size in world_sizes:
    plt.figure(figsize=(10, 6))
    subset = df[df['worldsize'] == world_size]
    plt.plot(subset['blocksize'], subset['time'], marker='o', linestyle='-', label=f'World Size: {world_size}')
    
    plt.title(f'Execution Time vs. Block Size for World Size {world_size}')
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.xticks(block_sizes)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'execution_time_vs_blocksize_{world_size}.png')
    plt.close()

# Plot cells updates per second for each configuration as a bar chart
plt.figure(figsize=(14, 8))
for i, block_size in enumerate(block_sizes):
    subset = df[df['blocksize'] == block_size].sort_values(by='worldsize')
    plt.bar(np.arange(len(world_sizes)) + i/len(block_sizes), subset['updates_per_second'],
            width=1/len(block_sizes), label=f'Block Size: {block_size}')

plt.title('Cells Updates Per Second by World Size and Block Size')
plt.xlabel('World Size')
plt.ylabel('Cells Updates Per Second')
plt.xticks(np.arange(len(world_sizes)) + 0.5 - 1/(2*len(block_sizes)), world_sizes)
plt.legend()
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('cells_updates_per_second.png')
plt.close()

print("Plots have been generated.")
