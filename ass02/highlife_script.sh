!/bin/bash

# Function to run the HighLife program with given arguments
run_highlife() {
    ./highlife "$1" "$2" "$3" "$4"
}
# 1024 2048 4096 8192 16384 32768
# Function to execute runs for different block sizes and world sizes
execute_runs() {
block_size=$1
iterations=1024

for world_size in 1024 2048 4096 8192 16384 32768 65536; do
	echo "Running HighLife with pattern 5, world size $world_size, $iterations iterations, and $block_size block_size."

        # Measure the execution time using the 'time' command
        real_time=$( { time run_highlife 5 "$world_size" "$iterations" "$block_size"; } 2>&1 | grep real | awk '{print $2}' )

        # Convert the real_time to seconds using 'bc'
        execution_time=$(echo "$real_time" | awk -F 'm|s' '{if ($2=="") print $1; else print $1*60+$2}' | bc)

        # Calculate cells updates per second using 'bc' for floating-point arithmetic
        cell_updates_per_sec=$(bc <<< "scale=2; $world_size * $world_size * $iterations / $execution_time")

        # Print the results
        echo "Execution Time: $execution_time seconds"
        echo "Cell Updates Per Second: $cell_updates_per_sec"
        echo ""

        # Save results to a CSV file
        echo "$block_size, $world_size, $execution_time, $cell_updates_per_sec" >> performance_results.csv
done
}

# Load modules and compile code on the front end
module load xl_r spectrum-mpi cuda
nvcc -O3 -gencode arch=compute_70,code=sm_70 highlife.cu -o highlife
# gcc highlife.c -o highlife
# Execute runs for different block sizes
execute_runs 8
execute_runs 16
execute_runs 32
execute_runs 128
execute_runs 512
execute_runs 102
