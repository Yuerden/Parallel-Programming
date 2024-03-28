//highlife.cu

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
    bool HL_kernelLaunch(unsigned char **d_data,
                         unsigned char **d_resultData,
                         size_t worldWidth,
                         size_t worldHeight,
                         int threadsCount,
                         int rank);
}

__global__ void HL_kernel(const unsigned char* d_data, 
                        unsigned int worldWidth, 
                        unsigned int worldHeight, 
                        unsigned char* d_resultData) 
{
    // Calculate the global index, this determines which cell each thread is responsible for.
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the stride, this is the total number of threads in the grid.
    // It is used to loop over the world in case there are more cells than threads.
    int stride = blockDim.x * gridDim.x;

    // Loop over the cells. Each thread may handle multiple cells if there are more cells than threads.
    for (int i = index; i < worldWidth * worldHeight; i += stride) {
        // Calculate the x and y coordinates of the cell.    
        int x = i % worldWidth;
        int y = i / worldWidth;

        // Compute the indices of the neighboring cells, with wrapping at edges.
        size_t x0 = (x + worldWidth - 1) % worldWidth;
        size_t x2 = (x + 1) % worldWidth;
        size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % worldHeight) * worldWidth;

        // Count the number of alive cells in the neighborhood.
        unsigned int aliveCells = d_data[x0 + y0] + d_data[x + y0] + d_data[x2 + y0] +
                                  d_data[x0 + y1] +                    d_data[x2 + y1] +
                                  d_data[x0 + y2] + d_data[x + y2] + d_data[x2 + y2];
        // Apply the rules of HighLife to determine if the cell will be alive or dead.
        // rule B36/S23: A cell is born if it has 3 neighbors, stays alive with 2 or 3 alive neighbors,
        // and dies otherwise. In HighLife, a dead cell with exactly 6 neighbors also becomes alive.        
        d_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !d_data[x + y1])
                                  || (aliveCells == 2 && d_data[x + y1]) ? 1 : 0;
    }
}

// helper function which lauches the kernel each iteration
bool HL_kernelLaunch(unsigned char** d_data, 
                    unsigned char** d_resultData, 
                    size_t worldWidth, 
                    size_t worldHeight, 
                    int threadsCount, 
                    int rank) 
{
   
    // Determine block and grid dimensions based on the number of threads
    dim3 blockDim(threadsCount, 1, 1);
    dim3 gridDim((worldWidth * worldHeight + blockDim.x - 1) / blockDim.x, 1, 1);

    //Set CUDA Device based on MPI rank.
    cudaError_t cE; // Declare the error variable
    int cudaDeviceCount; // Declare the device count variable
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess ) {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( rank % cudaDeviceCount )) != cudaSuccess ) {
        printf(" Unable to have rank %d set to cuda device %d, error is %d \n", rank, (rank % cudaDeviceCount), cE);
        exit(-1);
    }
    
    for (size_t i = 0; i < iterations; i++) {
        // Launch the CUDA kernel
        HL_kernel<<<gridDim, blockDim>>>(*d_data, worldWidth, worldHeight, *d_resultData);

        // Synchronize threads before swapping data
        cudaDeviceSynchronize();

        // Swap data pointers
        HL_swap(d_data, d_resultData);
    }

    return true;
}
