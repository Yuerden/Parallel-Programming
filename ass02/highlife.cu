#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world. 
unsigned char *g_data=NULL;

// Current width of world.
size_t g_worldWidth=0;

/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight

static inline void HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // calloc init's to all zeros
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    // Ensure memory is initialized to zero
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned_char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned_char));
}

static inline void HL_initAllOnes(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));

    for (int i = 0; i < g_dataLength; i++) {
        g_data[i] = 1;
    }

    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initOnesInMiddle(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char)); // Zero out the memory

    // Set a row of ones in the middle
    int start = 10 * g_worldWidth + 10;
    int end = 10 * g_worldWidth + 20;
    for (int i = start; i < end; i++) {
        g_data[i] = 1;
    }

    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initOnesAtCorners(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char)); // Zero out the memory

    // Set ones at the corners
    g_data[0] = 1;
    g_data[worldWidth - 1] = 1;
    g_data[(worldHeight - 1) * worldWidth] = 1;
    g_data[worldHeight * worldWidth - 1] = 1;

    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initSpinnerAtCorner(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char)); // Zero out the memory

    // Set spinner pattern at upper left corner
    g_data[0] = 1;
    g_data[1] = 1;
    g_data[worldWidth] = 1;

    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));
}

static inline void HL_initReplicator(size_t worldWidth, size_t worldHeight) {
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    // Allocate unified memory accessible by both CPU and GPU
    cudaMallocManaged(&g_data, g_dataLength * sizeof(unsigned char));
    cudaMallocManaged(&g_resultData, g_dataLength * sizeof(unsigned char));

    // Initially set all cells to dead
    cudaMemset(g_data, 0, g_dataLength * sizeof(unsigned char));
    cudaMemset(g_resultData, 0, g_dataLength * sizeof(unsigned char));

    // Calculate the starting point for the replicator pattern
    size_t centerX = worldWidth / 2;
    size_t centerY = worldHeight / 2;

    // Ensure the pattern fits in the world
    if (centerX >= 3 && centerY >= 3) {
        // Set cells for the replicator pattern
        size_t baseIndex = centerY * worldWidth + centerX;

        // Assuming a minimal replicator pattern
        g_data[baseIndex - worldWidth - 1] = 1;
        g_data[baseIndex - worldWidth] = 1;
        g_data[baseIndex - worldWidth + 1] = 1;

        g_data[baseIndex - 1] = 1;
        g_data[baseIndex + 1] = 1;

        g_data[baseIndex + worldWidth - 1] = 1;
        g_data[baseIndex + worldWidth] = 1;
        g_data[baseIndex + worldWidth + 1] = 1;
    }
}


static inline void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight )
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

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
  unsigned char *temp = *pA;
  *pA = *pB;
  *pB = temp;
}
 
static inline unsigned int HL_countAliveCells(unsigned char* data, 
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

/// Serial version of standard byte-per-cell life.
bool HL_iterateSerial(size_t iterations) 
{
  size_t i, y, x;

  for (i = 0; i < iterations; ++i) 
    {
      for (y = 0; y < g_worldHeight; ++y) 
	{
	  size_t y0 = ((y + g_worldHeight - 1) % g_worldHeight) * g_worldWidth;
	  size_t y1 = y * g_worldWidth;
	  size_t y2 = ((y + 1) % g_worldHeight) * g_worldWidth;
	  
	for (x = 0; x < g_worldWidth; ++x) 
	  {
	    size_t x0 = (x + g_worldWidth - 1) % g_worldWidth;
	    size_t x2 = (x + 1) % g_worldWidth;
	  
	    unsigned int aliveCells = HL_countAliveCells(g_data, x0, x, x2, y0, y1, y2);
	    // rule B36/S23
	    g_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !g_data[x + y1])
	      || (aliveCells == 2 && g_data[x + y1]) ? 1 : 0;
	}
      }
      HL_swap(&g_data, &g_resultData);

      // HL_printWorld(i);
    }
  
  return true;
}

__global__ void HL_kernel(const unsigned char* d_data, unsigned int worldWidth, unsigned int worldHeight, unsigned char* d_resultData) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; // Calculate the stride length

    // Iterate over the grid with a stride
    for (; index < worldWidth * worldHeight; index += stride) {
        int x = index % worldWidth;
        int y = index / worldWidth;

        // Calculate neighbor indices with wrap-around
        int x0 = (x + worldWidth - 1) % worldWidth;
        int x2 = (x + 1) % worldWidth;
        int y0 = (y + worldHeight - 1) % worldHeight * worldWidth;
        int y1 = y * worldWidth;
        int y2 = (y + 1) % worldHeight * worldWidth;

        // Apply HighLife rules
        unsigned int aliveCells = d_data[x0 + y0] + d_data[x + y0] + d_data[x2 + y0]
                                + d_data[x0 + y1] + d_data[x2 + y1]
                                + d_data[x0 + y2] + d_data[x + y2] + d_data[x2 + y2];
        bool isAlive = d_data[x + y1];
        bool nextState = (aliveCells == 3) || (aliveCells == 6 && !isAlive) || (aliveCells == 2 && isAlive);
        d_resultData[x + y1] = nextState ? 1 : 0;
    }
}


bool HL_kernelLaunch(unsigned char** d_data, unsigned char** d_resultData, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
    // Determine the number of blocks needed based on the world size and threads per block
    int blocksPerGrid = (worldWidth * worldHeight + threadsCount - 1) / threadsCount;
    
    for (size_t i = 0; i < iterationsCount; ++i) {
        // Launch the kernel with the dynamic number of threads per block
        HL_kernel<<<blocksPerGrid, threadsCount>>>(*d_data, worldWidth, worldHeight, *d_resultData);
        
        // Wait for GPU to finish before proceeding
        cudaDeviceSynchronize();
        
        // Swap the pointers to prepare for the next iteration
        HL_swap(d_data, d_resultData);
    }
    
    return true;
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <pattern> <worldSize> <iterations>\n", argv[0]);
        return -1;
    }

    unsigned int pattern = (unsigned int)atoi(argv[1]);
    size_t worldSize = (size_t)atoi(argv[2]);
    size_t iterations = (size_t)atoi(argv[3]);

    // Initialize the world with the specified pattern
    HL_initMaster(pattern, worldSize, worldSize);

    // Launch the CUDA kernel to process the world for the given number of iterations
    if (!HL_kernelLaunch(&g_data, &g_resultData, worldSize, worldSize, iterations)) {
        fprintf(stderr, "Error launching HighLife CUDA kernel\n");
        return -1;
    }

    // Optionally, print the final state of the world
    // HL_printWorld(iterations); // Make sure this function is adapted for CUDA if used

    // Free the allocated memory
    cudaFree(g_data);
    cudaFree(g_resultData);

    return 0;
}

