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
    
    // Variables for the simulation
    int myrank, numranks;
    int rootrank = 0;
    unsigned int pattern, world_N, iterations;
    double startTime, endTime;

    // Check command-line arguments
    if (argc != 4) {
        // if (myrank == 0) {
        printf("Usage: mpirun -np <num_processes> ./highlife <pattern> <world_size> <iterations>\n");
        // }
        // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); //CANT ABORT IF MPI SETUP HASNT HAPPENED YET DUMMY
        return 0;
    }

    pattern = atoi(argv[1]);
    world_N = atoi(argv[2]);
    iterations = atoi(argv[3]);
    //printf("Pattern: %u, World_N: %u, Iterations: %u\n", pattern, world_N, iterations);

    // Setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    
    // Calculate chunk stuff
    int chunkHeight = world_N / numranks;
    int chunkSize = world_N * chunkHeight;
    unsigned char *chunk_data=NULL;
    unsigned char *ghostAbove=NULL;
    unsigned char *ghostBellow=NULL;


    // Initialization specific for Rank 0
    if (myrank == rootrank) {
        startTime = MPI_Wtime();

        // Initialize a local copy of the whole NxN Cell World to global g_data at top of program
        HL_initMaster(pattern, world_N, world_N);
        
        // Allocation "chunk" of Cell Universe plus "ghost" rows for RootRank
        chunk_data = calloc(chunkSize, sizeof(unsigned char));

        // Scatter the initialized world to all ranks
        // Since Rank 0 also needs a part of the world, use MPI_Scatter with special handling
        unsigned char* scatterBuffer = malloc(world_N * world_N * sizeof(unsigned char)); // Temporary buffer for scattering
        memcpy(scatterBuffer, g_data, world_N * world_N * sizeof(unsigned char)); // Copy initialized world from global data pointer into buffer
        /*int MPI_Scatter(void *sendbuf, int sendcount, 
			MPI_Datatype senddatatype, void *recvbuf, 
			int recvcount, MPI_Datatype recvdatatype, 
			int source, MPI_Comm comm) */
        MPI_Scatter(scatterBuffer, chunkSize, MPI_UNSIGNED_CHAR,
                    chunk_data, chunkSize, MPI_UNSIGNED_CHAR, rootrank, MPI_COMM_WORLD);
        free(scatterBuffer);
    } else {
        // Allocation "slice" of universe plus "ghost" rows;
        chunk_data = calloc(chunkSize, sizeof(unsigned char));
        MPI_Scatter(NULL, chunkSize, MPI_UNSIGNED_CHAR, 
                    chunk_data, chunkSize, MPI_UNSIGNED_CHAR, rootrank, MPI_COMM_WORLD);
    }

    // Main simulation loop
    int rowLength = world_N;
    unsigned char *myChunk_TopRow_Buffer = malloc(rowLength * sizeof(unsigned char)); // Temporary buffer for scattering
    unsigned char *myChunk_BottomRow_Buffer = malloc(rowLength * sizeof(unsigned char)); // Temporary buffer for scattering
    int prev_rank, next_rank;
    unsigned char *myTopGhost = malloc(rowLength * sizeof(unsigned char));
    unsigned char *myBottomGhost = malloc(rowLength * sizeof(unsigned char));
    for (int i = 0; i < iterations; i++) {        
        //MY EXCHANGE::
        //-MPI Variable initalization
        MPI_Request send_requests[2], recv_requests[2];
        //-Get this top and bottom row data into respective buffers
        memcpy(myChunk_TopRow_Buffer, chunk_data, rowLength * sizeof(unsigned char));

        memcpy(myChunk_BottomRow_Buffer, chunk_data+((chunkHeight-1)*rowLength), rowLength * sizeof(unsigned char));
        //-Send these to previous and next ranks
        prev_rank = (myrank - 1 + numranks) % numranks;
        next_rank = (myrank + 1) % numranks;
        MPI_Isend(myChunk_TopRow_Buffer, rowLength, MPI_UNSIGNED_CHAR, 
            prev_rank, 0, MPI_COMM_WORLD, &send_requests[0]); //my top goes to prev bottom ghost
        MPI_Isend(myChunk_BottomRow_Buffer, rowLength, MPI_UNSIGNED_CHAR, 
            next_rank, 1, MPI_COMM_WORLD, &send_requests[1]); //my bottom goes to next top ghost
        //-Retrive top and bottom ghosts
        MPI_Irecv(myTopGhost, rowLength, MPI_UNSIGNED_CHAR, 
            prev_rank, 1, MPI_COMM_WORLD, &recv_requests[0]); //my top ghost from prev
        MPI_Irecv(myBottomGhost, rowLength, MPI_UNSIGNED_CHAR, 
            next_rank, 0, MPI_COMM_WORLD, &recv_requests[1]); //my bottom ghost from next
        //-Wait All
        MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);
        MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);

        //MY SERIAL:
        int y, x;
        unsigned int aliveCells;
        size_t y0, y1, y2, x0, x2;
        for (y = 0; y < chunkHeight; ++y)
        {
            y0 = (y - 1) * rowLength;
            y1 = y * rowLength;
            y2 = (y + 1) * rowLength;
            
            for (x = 0; x < rowLength; ++x)
            {
                x0 = (x + rowLength - 1) % rowLength;
                x2 = (x + 1) % rowLength;

                //aliveCells:
                //if y = 0, y0 now uses top ghost
                if(y==0){
                    aliveCells = myTopGhost[x0] + myTopGhost[x] + myTopGhost[x2]
                               + chunk_data[x0 + y1]                       + chunk_data[x2 + y1]
                               + chunk_data[x0 + y2] + chunk_data[x + y2] + chunk_data[x2 + y2];

                }
                //if y = chunkHeight-1, y2 uses bottom ghost
                else if(y==chunkHeight-1){
                    aliveCells = chunk_data[x0 + y0] + chunk_data[x + y0] + chunk_data[x2 + y0]
                               + chunk_data[x0 + y1]                       + chunk_data[x2 + y1]
                               + myBottomGhost[x0] + myBottomGhost[x] + myBottomGhost[x2];

                }else{
                    aliveCells = chunk_data[x0 + y0] + chunk_data[x + y0] + chunk_data[x2 + y0]
                               + chunk_data[x0 + y1]                       + chunk_data[x2 + y1]
                               + chunk_data[x0 + y2] + chunk_data[x + y2] + chunk_data[x2 + y2];

                }

                g_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !chunk_data[x + y1]) || (aliveCells == 2 && chunk_data[x + y1]) ? 1 : 0;
            }
        }
        HL_swap(&chunk_data, &g_resultData);
    }
    free(myChunk_TopRow_Buffer);
    free(myChunk_BottomRow_Buffer);
    free(myTopGhost);
    free(myBottomGhost);
    // Synchronize before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    // Rank 0 calculates and prints the execution time
    if (myrank == 0) {
        endTime = MPI_Wtime();
        printf("Simulation time: %f seconds\n", endTime - startTime);
    }

    bool print = true;
    if(print){
        // Gather  fithenal world state at Rank 0 for output
        unsigned char* finalWorld = NULL;
        // Allocate memory to receive the final world state
        if(myrank==rootrank) {
            finalWorld = malloc(world_N * world_N * sizeof(unsigned char));
            MPI_Gather(chunk_data, chunkSize, MPI_UNSIGNED_CHAR,
                    finalWorld, chunkSize, MPI_UNSIGNED_CHAR, rootrank, MPI_COMM_WORLD);
        }else{
            MPI_Gather(chunk_data, chunkSize, MPI_UNSIGNED_CHAR,
                    NULL, 0, MPI_UNSIGNED_CHAR, rootrank, MPI_COMM_WORLD);
        }
        if(myrank==rootrank)
        {
            unsigned char* temp = g_data;
            g_data = finalWorld;
            HL_printWorld(iterations); // Assuming iterations is the desired final iteration count
            // Reset the g_data pointer and free the final world memory
            g_data = temp;
            free(finalWorld);
        }
    }

    // Finalize MPI and clean up...
    free(g_data);
    free(g_resultData);
    free(chunk_data);
    MPI_Finalize();
    return 0;
}