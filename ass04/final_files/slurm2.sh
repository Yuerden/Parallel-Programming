#!/bin/bash -x
#SBATCH -t 00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:6
#SBATCH -o joboutput.%J

module load xl_r spectrum-mpi cuda/11.2

# Function to execute runs for different configurations
execute_runs() {
    num_procs=$1
    pattern=$2
    world_size=$3
    iterations=$4
    block_size=$5

    echo "Running HighLife with $num_procs processes, pattern $pattern, world size $world_size, $iterations iterations, and $block_size block size."

    # Run the highlife-exe program and capture the output
    output=$(mpirun -np "$num_procs" ./highlife-exe "$pattern" "$world_size" "$iterations" "$block_size" | tee /dev/tty)

    # Extract the execution time from the output
    execution_time=$(echo "$output" | grep "Performance time:" | awk '{print $3}')

    # Calculate cells updates per second using the execution time
    cell_updates_per_sec=$(bc <<< "scale=2; $world_size * $world_size * $iterations / $execution_time")

    # Print the results
    echo "Execution Time: $execution_time seconds"
    echo "Cell Updates Per Second: $cell_updates_per_sec"
    echo ""

    # Save results to a CSV file
    echo "$execution_time, $cell_updates_per_sec" >> performance_results2.csv
}

# Compile your highlife-exe program here if needed
# nvcc ... or gcc ...

# Execute runs for one configuration (12 GPU/MPI ranks)
execute_runs 12 5 16384 128 256
