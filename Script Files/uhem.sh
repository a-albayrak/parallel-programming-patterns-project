#!/bin/bash
#SBATCH --job-name=vtune_hotspot_generic     # Job name
#SBATCH --output=vtune_hotspot_%j.log       # Output log file
#SBATCH --nodes=1                           # Number of nodes to use
#SBATCH --ntasks=1                          # Number of tasks
#SBATCH --cpus-per-task=16                  # Maximum number of CPUs (up to 16 threads will be used)
#SBATCH --time=01:00:00                     # Maximum runtime
#SBATCH --partition=defq                    # Partition name changed to defq (instead of shortq)

# Load the VTune module
module load intel/vtune_amplifier_xe_2019u4

# Path to the program to be analyzed
PROGRAM_PATH="./strassen_UHEM"

# Thread counts
THREAD_COUNTS=(2 4 8 16)

# Matrix sizes
MATRIX_SIZES=(1024 2048 4096)

# VTune Hotspot Analysis
for THREAD_COUNT in "${THREAD_COUNTS[@]}"; do
    for MATRIX_SIZE in "${MATRIX_SIZES[@]}"; do
        echo "Running $PROGRAM_PATH with $THREAD_COUNT threads and matrix size $MATRIX_SIZE..."
        export OMP_NUM_THREADS=$THREAD_COUNT  # Set the number of OpenMP threads
        
        amplxe-cl -collect hotspots -r vtune_${THREAD_COUNT}_${MATRIX_SIZE}_$(basename $PROGRAM_PATH) $PROGRAM_PATH $MATRIX_SIZE
    done
done
