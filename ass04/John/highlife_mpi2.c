#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<mpi.h>

extern bool HL_kernelLaunch(unsigned char **d_data,
                         unsigned char **d_resultData,
                         size_t worldWidth,
                         size_t worldHeight,
                         int threadsCount,
                         int rank);

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


int main(int argc, char *argv[])
{
    // initialize MPI
    int myrank, numranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Request request;

    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;
    int threadCount = 0;

    // printf("This is the HighLife running in serial on a CPU.\n");

    if( argc != 5 )
    {
	printf("HighLife requires 3 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations, e.g. ./highlife 0 32 2 \n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threadCount = atoi(argv[4]);
    unsigned char* chunk = NULL;
    unsigned char* plus_chunk = NULL;
    unsigned char* res_chunk = NULL;
    unsigned char* top_ghost = NULL;
    unsigned char* bottom_ghost = NULL;
    // MPI implementation
    double start, end;
    int chunksize = worldSize * worldSize;
    int plusRows = worldSize + 2;
    int plusSize = plusRows * worldSize;
    res_chunk = calloc(plusSize, sizeof(unsigned char));
    if(myrank == 0)
    {
        // start time
        start = MPI_Wtime();
        // initialize local NxN world
        HL_initMaster(pattern, worldSize, worldSize);
        //HL_printWorld(0);
    }
    // allocation chunk of universe plus ghost rows
    chunk = calloc(plusSize, sizeof(unsigned char));
    top_ghost = calloc(worldSize, sizeof(unsigned char));
    bottom_ghost = calloc(worldSize, sizeof(unsigned char));
    
    int i;
    for(i = 0; i < chunksize; i++){
        chunk[i+worldSize] = g_data[i];
    }
    // perform iterations
    unsigned char* top_send = calloc(worldSize, sizeof(unsigned char));
    unsigned char* bottom_send = calloc(worldSize, sizeof(unsigned char));
    for(i=0;i<iterations;i++)
    {
        //printf("rank: %d\t iter: %d\n", myrank, i);
        // populate top_send and bottom_send
        int j;
        for(j=0;j<worldSize;j++)
        {
            if(chunk == NULL)
                printf("ERR: chunk w/ rank %d null\n", myrank);
            top_send[j] = chunk[j+worldSize];
            bottom_send[j] = chunk[plusSize - worldSize - worldSize + j];
        }
        /*
        if(i==0)
        {
            printf("top_send, bottom_send\n");
            for(j=0;j<worldSize;j++)
                printf("%u ", (unsigned char)top_send[j]);
            printf("\n");
            for(j=0;j<worldSize;j++)
                printf("%u ", (unsigned char)bottom_send[j]);
            printf("\n");
        }
        */
        // exchange ghost row data w/ MPI ranks using MPI_Isend/Irecv
        // if rank is even: send then receive
        if(myrank % 2 == 0)
        {
            // send 1st row of chunk to bottom_ghost of last chunk
            if(myrank == 0)
            {
                MPI_Isend(bottom_send, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 1, MPI_COMM_WORLD, &request);
                MPI_Irecv(top_ghost, worldSize, MPI_UNSIGNED_CHAR, numranks-1, 2, MPI_COMM_WORLD, &request);
                MPI_Isend(top_send, worldSize, MPI_UNSIGNED_CHAR, numranks-1, 3, MPI_COMM_WORLD, &request);
                MPI_Irecv(bottom_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 4, MPI_COMM_WORLD, &request);
            }
            else
            {
                MPI_Isend(bottom_send, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 1, MPI_COMM_WORLD, &request);
                MPI_Irecv(top_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 2, MPI_COMM_WORLD, &request);
                MPI_Isend(top_send, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 3, MPI_COMM_WORLD, &request);
                MPI_Irecv(bottom_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 4, MPI_COMM_WORLD, &request);
            }
        }
        // if rank is odd: receive then send
        else
        {
            if(myrank == numranks-1)
            {
                MPI_Irecv(top_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 1, MPI_COMM_WORLD, &request);
                MPI_Isend(bottom_send, worldSize, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD, &request);
                MPI_Irecv(bottom_ghost, worldSize, MPI_UNSIGNED_CHAR, 0, 3, MPI_COMM_WORLD, &request);
                MPI_Isend(top_send, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 4, MPI_COMM_WORLD, &request);
            }
            else
            {
                MPI_Irecv(top_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 1, MPI_COMM_WORLD, &request);
                MPI_Isend(bottom_send, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 2, MPI_COMM_WORLD, &request);
                MPI_Irecv(bottom_ghost, worldSize, MPI_UNSIGNED_CHAR, myrank+1, 3, MPI_COMM_WORLD, &request);
                MPI_Isend(top_send, worldSize, MPI_UNSIGNED_CHAR, myrank-1, 4, MPI_COMM_WORLD, &request);
            }
        }
        // barrier call to ensure ghost rows update properly
        MPI_Barrier(MPI_COMM_WORLD);
        //copy recieved rows back into chunk
        for(j = 0; j<;j++){
            chunk[j]=top_ghost[j];
            chunk[j+plusSize-worldSize]=bottom_ghost[j];
        }
        //KERNALLEALLA!!!!!!!!!!
        HL_kernelLaunch(&chunk, &res_chunk, worldSize, plusRows, threadCount, myrank);

        HL_swap(&chunk, &res_chunk);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(myrank == 0)
    {
        // end time and printf MPI_Wtime
        end = MPI_Wtime();
        printf("Total execution time: %f s\n", end-start);
        // performance results
    }

    //copy back from chunk->g_data
    for(i = 0; i < chunksize; i++){
        g_data[i] = chunk[i+worldSize];
    }
    printf("Rank: %d\n", rank);
    HL_printWorld(iterations);

    //if(myrank == 0)
    //    HL_printWorld(iterations);
    free(chunk);
    free(res_chunk);
    free(top_ghost);
    free(bottom_ghost);
    free(top_send);
    free(bottom_send);
    free(g_data);
    free(g_resultData);
    MPI_Finalize();

    /*
    printf("AFTER INIT IS............\n");
    HL_printWorld(0);
    
    HL_iterateSerial( iterations );
    
    printf("######################### FINAL WORLD IS ###############################\n");
    HL_printWorld(iterations);
    */
    
    //return true;
}
