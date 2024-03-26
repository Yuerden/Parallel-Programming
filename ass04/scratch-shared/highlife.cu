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

__global__ void HL_kernel(const unsigned char *d_data,
                          size_t worldWidth,
                          size_t worldHeight,
                          unsigned char *d_resultData)
{
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gridSize = worldWidth * worldHeight;

    for (; index < gridSize; index += blockDim.x * gridDim.x)
    {
        // Calculate x and y coordinates from the flattened index
        size_t x = index % worldWidth;
        size_t y = index / worldWidth;

        // Calculate the indices of neighboring cells
        size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % worldHeight) * worldWidth;
        size_t x0 = (x + worldWidth - 1) % worldWidth;
        size_t x2 = (x + 1) % worldWidth;

        // Check neighboring cells to determine how many are currently alive
        unsigned int aliveCells = d_data[x0 + y0] + d_data[x + y0] + d_data[x2 + y0] +
                                  d_data[x0 + y1] +                    d_data[x2 + y1] +
                                  d_data[x0 + y2] + d_data[x + y2] + d_data[x2 + y2];

        // Assign the next state of the cell based on the number of alive neighbors
        d_resultData[y1 + x] = (aliveCells == 3) || (aliveCells == 6 && !d_data[y1 + x]) ||
                                (aliveCells == 2 && d_data[y1 + x]) ? 1 : 0;
    }
}


// helper function which lauches the kernel each iteration
bool HL_kernelLaunch(unsigned char **d_data,
                                unsigned char **d_resultData,
                                size_t worldWidth,
                                size_t worldHeight,
                                int threadsCount,
                                int rank)
{
    cudaSetDevice( rank );
    // determine num blocks by roughly dividing the array size by the number of threads
    dim3 blocks = dim3((worldHeight * worldWidth - 1 + threadsCount) / threadsCount, 1, 1);
    // three dimensional variable for number of threads
    dim3 threads = dim3(threadsCount, 1, 1);
    // launch the kernel with the given block and thread number, each thread will operate on one element
    HL_kernel<<<blocks, threads>>>(*d_data, worldWidth, worldHeight, *d_resultData);
    // ensure all threads have finished their calculations
    cudaDeviceSynchronize();

    return true;
}