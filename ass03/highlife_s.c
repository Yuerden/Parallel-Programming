#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>
#include<string.h>

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
    g_data = calloc( g_dataLength, sizeof(unsigned char));
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initAllOnes( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set all rows of world to true
    for( i = 0; i < g_dataLength; i++)
    {
	    g_data[i] = 1;
    }
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight )
{
    int i;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    // set first 1 rows of world to true
    for( i = 10*g_worldWidth; i < 11*g_worldWidth; i++)
    {
	    if( (i >= ( 10*g_worldWidth + 10)) && (i < (10*g_worldWidth + 20)))
	    {
	        g_data[i] = 1;
	    }
    }
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[worldWidth-1]=1; // upper right
    g_data[(worldHeight * (worldWidth-1))]=1; // lower left
    g_data[(worldHeight * (worldWidth-1)) + worldWidth-1]=1; // lower right
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight )
{
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    g_data[0] = 1; // upper left
    g_data[1] = 1; // upper left +1
    g_data[worldWidth-1]=1; // upper right
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    size_t x, y;
    
    g_worldWidth = worldWidth;
    g_worldHeight = worldHeight;
    g_dataLength = g_worldWidth * g_worldHeight;

    g_data = calloc( g_dataLength, sizeof(unsigned char));

    x = worldWidth/2;
    y = worldHeight/2;
    
    g_data[x + y*worldWidth + 1] = 1; 
    g_data[x + y*worldWidth + 2] = 1;
    g_data[x + y*worldWidth + 3] = 1;
    g_data[x + (y+1)*worldWidth] = 1;
    g_data[x + (y+2)*worldWidth] = 1;
    g_data[x + (y+3)*worldWidth] = 1; 
    
    g_resultData = calloc( g_dataLength, sizeof(unsigned char)); 
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
                g_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !g_data[x + y1]) || (aliveCells == 2 && g_data[x + y1]) ? 1 : 0;
            }
        }
        HL_swap(&g_data, &g_resultData);

        // HL_printWorld(i);
    }

    return true;
}

// Updated function to exchange ghost rows between MPI ranks correctly
static inline void exchangeGhostRows(int myrank, int numranks, size_t realHeight, size_t totalWidth) {
    MPI_Request send_requests[2], recv_requests[2];
    int prev_rank = (myrank - 1 + numranks) % numranks;
    int next_rank = (myrank + 1) % numranks;

    // Correct exchange logic
    MPI_Isend(g_data + totalWidth, totalWidth, MPI_UNSIGNED_CHAR, prev_rank, 0, MPI_COMM_WORLD, &send_requests[0]); // Send top real row to prev
    MPI_Isend(g_data + totalWidth * (realHeight - 2), totalWidth, MPI_UNSIGNED_CHAR, next_rank, 1, MPI_COMM_WORLD, &send_requests[1]); // Send bottom real row to next

    MPI_Irecv(g_data, totalWidth, MPI_UNSIGNED_CHAR, prev_rank, 1, MPI_COMM_WORLD, &recv_requests[0]); // Receive top ghost row from prev
    MPI_Irecv(g_data + totalWidth * (realHeight - 1), totalWidth, MPI_UNSIGNED_CHAR, next_rank, 0, MPI_COMM_WORLD, &recv_requests[1]); // Receive bottom ghost row from next

    MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);
}


int main(int argc, char *argv[]) {
    int myrank, numranks;
    unsigned int pattern, worldSize, iterations;
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    if (argc != 4) {
        if (myrank == 0) {
            printf("Usage: mpirun -np <num_processes> ./highlife <pattern> <world_size> <iterations>\n");
        }
        MPI_Finalize();
        return -1;
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);

    size_t chunkHeight = worldSize / numranks;
    size_t realHeight = chunkHeight + 2; // Including ghost rows
    g_worldWidth = worldSize;

    unsigned char* scatterBuffer = NULL;
    if (myrank == 0) {
        HL_initMaster(pattern, worldSize, worldSize);
        // Only allocate scatter buffer in rank 0
        scatterBuffer = malloc(worldSize * worldSize * sizeof(unsigned char));
        memcpy(scatterBuffer, g_data, worldSize * worldSize * sizeof(unsigned char));
        startTime = MPI_Wtime();
    }

    // Allocate only the necessary space in each rank
    g_dataLength = g_worldWidth * realHeight;
    g_data = calloc(g_dataLength, sizeof(unsigned char));
    g_resultData = calloc(g_worldWidth * chunkHeight, sizeof(unsigned char));

    // Correctly distribute the work to all ranks
    MPI_Scatter(scatterBuffer, g_worldWidth * chunkHeight, MPI_UNSIGNED_CHAR, g_data + g_worldWidth, g_worldWidth * chunkHeight, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Main simulation loop with corrected exchange of ghost rows and iteration
    for (int i = 0; i < iterations; i++) {
        exchangeGhostRows(myrank, numranks, realHeight, g_worldWidth);
        HL_iterateSerial(1); // Modify this function if necessary to correctly handle edges with ghost row data
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
        endTime = MPI_Wtime();
        printf("Simulation time: %f seconds\n", endTime - startTime);
    }

    // Gather results back to rank 0
    if (myrank == 0) {
        scatterBuffer = realloc(scatterBuffer, worldSize* worldSize * sizeof(unsigned char)); // Ensure enough space is allocated for gathering
    }
    MPI_Gather(g_data + g_worldWidth, g_worldWidth * chunkHeight, MPI_UNSIGNED_CHAR, scatterBuffer, g_worldWidth * chunkHeight, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        // Replace g_data with the gathered full world for output
        free(g_data);
        g_data = scatterBuffer;
        printf("######################### FINAL WORLD IS ###############################\n");
        HL_printWorld(iterations);
    }

    free(g_data);
    free(g_resultData);
    MPI_Finalize();
    return 0;
}

// int main(int argc, char *argv[]) {
//     //Setup MPI
//         //MPI init stuff
//         // MPI_Init(&argc, &argv);
//         // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//         // MPI_Comm_size(MPI_COMM_WORLD, &numranks);
//     int myrank, numranks;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//     MPI_Comm_size(MPI_COMM_WORLD, &numranks);

//     if (argc != 4) {
//         if (myrank == 0) {
//             printf("Usage: mpirun -np <num_processes> ./highlife <pattern> <world_size> <iterations>\n");
//         }
//         MPI_Finalize();
//         return -1;
//     }
//     unsigned int pattern, worldSize, iterations;
//     pattern = atoi(argv[1]);
//     worldSize = atoi(argv[2]);
//     iterations = atoi(argv[3]);    

//     double startTime, endTime;
//     if(rank == 0){
//         //start time with MPI_Wtime;
//         startTime = MPI_Wtime();

//         //init a local copy of NxN Cell World;
//         //Allocation "chunk" of Cell Universe plus "ghost" rows;

//         //MPI_Scatter NxN Cell World with Rank 0 as base.
//     }
//     else
//     {
//         //Allocation "slice" of universe plus "ghost" rows;

//         //MPI_Scatter with Rank 0 as base and recv into your local universe chunk/slice

//     }

//     for(i=0; i< ticks; i++)
//     {
//         //Exchange "ghost" row data with MPI Ranks using Isend/Irecv

//         //Do rest of universeupdate as done in ass1 using serial HighLife Functions
//             //note, no top-bottom wrap except for ranks 0 and N-1.
//     }

//     //MPI_Barrier();

//     if(rank == 0)
//     {
//         //end time with MPI_Wtime and printf MPI_Wtime performance results;
//     }

//     if(true)
//     {
//         //Use MPI_Gather to assemble Cell Universe to Rank 0
//             // this will used for small "test" worlds only - < 64x64
        
//         //Output world using serial function provide in template.
//     }
    
//     //MPI_Finalize();
//     free(g_data);
//     free(g_resultData);
//     MPI_Finalize();
//     return 0;
// }