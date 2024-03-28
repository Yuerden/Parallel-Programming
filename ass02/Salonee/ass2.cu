// highlife.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Global variables
unsigned char *g_resultData = NULL;
unsigned char *g_data = NULL;
size_t g_worldWidth = 0;
size_t g_worldHeight = 0;
size_t g_dataLength = 0;

static inline void HL_initAllZeros(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Use cudaMallocManaged for memory allocation
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));

    // Initialize to zeros
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initAllOnes(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Use cudaMallocManaged for memory allocation
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));

    // Initialize to ones
    cudaMemset(g_data, 1, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 1, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight)
{
     int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    //g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set first 1 rows of world to true
    for( i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
	if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
	{
	    g_data[i] = 1;
	}
    }
    
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    //g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 


}

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    //g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[worldWidth-1]=1; // upper right
    g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char)); 
    //g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    //g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[1] = 1; // upper left +1
    g_data[worldWidth-1]=1; // upper right
    
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char)); 
    //g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    size_t x, y;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    //g_data = calloc( g_dataLength, sizeof(unsigned char));

    x = worldWidth/2;
    y = worldHeight/2;
    
    g_data[x + y*worldWidth + 1] = 1; 
    g_data[x + y*worldWidth + 2] = 1;
    g_data[x + y*worldWidth + 3] = 1;
    g_data[x + (y+1)*worldWidth] = 1;
    g_data[x + (y+2)*worldWidth] = 1;
    g_data[x + (y+3)*worldWidth] = 1; 
    
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char)); 
    //g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

inline void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
{
    switch(pattern)
    {
    case 0:
	HL_initAllZeros( worldWidth, worldHeight );
	break;
	
    case 1:
	HL_initAllOnes( worldWidth, worldHeight );
	break;
	
    case 2:
	HL_initOnesInMiddle( worldWidth, worldHeight );
	break;
	
    case 3:
	HL_initOnesAtCorners( worldWidth, worldHeight );
	break;

    case 4:
	HL_initSpinnerAtCorner( worldWidth, worldHeight );
	break;

    case 5:
	HL_initReplicator( worldWidth, worldHeight );
	break;
	
    default:
	printf("Pattern %u has not been implemented \n", pattern);
	exit(-1);
    }
}

// Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);
    
    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
      }

    printf("\n\n");
}



static inline void HL_swap(unsigned char **pA, unsigned char **pB) {
    unsigned char *temp = *pA;
    *pA = *pB;
    *pB = temp;
}

static inline __device__  unsigned int HL_countAliveCells(const unsigned char* data, 
					   size_t x0, 
					   size_t x1, 
					   size_t x2, 
					   size_t y0, 
					   size_t y1,
					   size_t y2) 
{
  
  return data[x0 + y0] + data[x1 + y0] + data[x2 + y0]
    + data[x0 + y1] + data[x2 + y1]
    + data[x0 + y2] + data[x1 + y2] + data[x2 + y2];
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
        unsigned int aliveCells = HL_countAliveCells(d_data, x0, x, x2, y0, y1, y2);
        // Apply the rules of HighLife to determine if the cell will be alive or dead.
        // rule B36/S23: A cell is born if it has 3 neighbors, stays alive with 2 or 3 alive neighbors,
        // and dies otherwise. In HighLife, a dead cell with exactly 6 neighbors also becomes alive.        
        d_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !d_data[x + y1])
                                  || (aliveCells == 2 && d_data[x + y1]) ? 1 : 0;
    }
}

bool HL_kernelLaunch(unsigned char** d_data, 
                    unsigned char** d_resultData, 
                    size_t worldWidth, 
                    size_t worldHeight, 
                    size_t iterations, 
                    ushort threadsCount) 
{
   
    // Determine block and grid dimensions based on the number of threads
    dim3 blockDim(threadsCount, 1, 1);
    dim3 gridDim((worldWidth * worldHeight + blockDim.x - 1) / blockDim.x, 1, 1);

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


int main(int argc, char *argv[])
{
    unsigned int  pattern = 0;
    size_t worldSize = 0;
    size_t iterations = 0;
    ushort thread_blksize = 0; 

    // printf("This is the HighLife running in serial on a CPU.\n");

    if( argc != 5 )
    {
    printf("HighLife requires 4 arguments, 1st is pattern number, 2nd the sq size of the world, 3rd is the number of iterations, and fourth is the block size, e.g. ./highlife 0 32 2 32 \n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    thread_blksize = atoi(argv[4]);
    
    HL_initMaster(pattern, worldSize, worldSize);
    // printf("AFTER INIT IS............\n");
    // HL_printWorld(0);
    
    // Call the parallel kernel
    HL_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, iterations, thread_blksize);
    
    // printf("######################### FINAL WORLD IS ###############################\n");
    HL_printWorld(iterations);

    cudaFree(g_data);
    cudaFree(g_resultData);
    
    return true;
}
