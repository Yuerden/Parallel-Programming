//scratch-shared>highlife-mpi.c>:
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

static inline void HL_initOnesInMiddle( size_t worldWidth, size_t worldHeight ){
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

static inline void HL_initOnesAtCorners( size_t worldWidth, size_t worldHeight)
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

static inline void HL_initSpinnerAtCorner( size_t worldWidth, size_t worldHeight)
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

int main(int argc, char *argv[])
{
    unsigned int pattern = atoi(argv[1]);
    unsigned int worldSize = atoi(argv[2]);
    unsigned int iterations = atoi(argv[3]);
    
    int numranks, myrank;
    double start_time, end_time;

    // setup MPI
    MPI_Status status;
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &numranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    int rows_per_rank = worldSize / numranks
    unsigned int cells_per_rank = rows_per_rank * worldSize;
    // printf("cells_per_rank = %u\n",cells_per_rank);

    // request and status arrays to make sure isend/irecv completes
    MPI_Request request_arr[4];
    MPI_Status status_arr[4];
    int request_index = 0;

    if (myrank == 0) {
        start_time = MPI_Wtime();
        // init local copy of whole NxN Cell World
        HL_initMaster(pattern, worldSize, worldSize);
        // Allocation "chunk" of Cell Universe plus "ghost" rows
        // HL_printWorld(0);

    }
        // MPI_Scatter Nxn Cell World with Rank 0 as base
        // Allocation "slice" of universe plus "ghost" rows;
        // MPI_Scatter with Rank 0 as base and recv into your local universe chunk/slice

    else {
        g_dataLength = worldSize * worldSize;
        g_data = calloc( g_dataLength, sizeof(unsigned char));
        g_resultData = calloc( g_dataLength, sizeof(unsigned char));
    }

    // INCLUDES THE GHOST ROWS
    unsigned char * sub_world = calloc(cells_per_rank + worldSize*2, sizeof(unsigned char));
    unsigned char * sub_results = calloc(cells_per_rank + worldSize*2, sizeof(unsigned char));
    
    MPI_Scatter(g_data,cells_per_rank,MPI_UNSIGNED_CHAR,sub_world+worldSize,cells_per_rank,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    unsigned char * send_top_row = calloc(worldSize, sizeof(unsigned char));
    unsigned char * send_bottom_row = calloc(worldSize, sizeof(unsigned char));
    
    int i,j,k,l,m;
    for (i = 0; i < iterations; i++) {
        // printf("iteration: %d\n",i);
        for (j = 0, k = worldSize; j < worldSize; j++, k++) {
            send_top_row[j] = sub_world[k];
        }
        for (l = worldSize-1, m = worldSize + cells_per_rank - 1; l >= 0; ; l--, m--) {
            send_bottom_row[l] = sub_world[m];
        }

        // Exchange "ghost" row data with MPI Ranks using MPI_Isend/Irecv
        // no top-bottom wrap except for ranks 0 and N-1
        MPI_Irecv(sub_world,worldSize,MPI_UNSIGNED_CHAR,(myrank - 1 + numranks) % numranks, 0, MPI_COMM_WORLD, &request_arr[request_index++]);
        MPI_Isend(send_bottom_row,worldSize,MPI_UNSIGNED_CHAR,(myrank + 1) % numranks, 0, MPI_COMM_WORLD,&request_arr[request_index++]);
        
        MPI_Irecv(sub_world+worldSize+cells_per_rank,worldSize,MPI_UNSIGNED_CHAR,(myrank + 1) % numranks, 0, MPI_COMM_WORLD, &request_arr[request_index++]);
        MPI_Isend(send_top_row,worldSize,MPI_UNSIGNED_CHAR,(myrank - 1 + numranks) % numranks, 0, MPI_COMM_WORLD,&request_arr[request_index++]);

        MPI_Waitall(4,request_arr,status_arr);

        memcpy(sub_results,sub_world,cells_per_rank + worldSize*2);


        // Do rest of universe update using serial HighLife functions
        size_t y, x;
        // Because of ghost rows do not need % operator for wrap around
        for (y = 1; y <= rows_per_rank; ++y) {
            size_t y0 = (y-1) * worldSize;
            size_t y1 = y * worldSize;
            size_t y2 = (y + 1) * worldSize;

            for (x = 0; x < worldSize; ++x) {
                size_t x0 = (x + worldSize - 1) % worldSize;
                size_t x2 = (x + 1) % worldSize;
                
                unsigned int aliveCells = sub_world[x0 + y0] + sub_world[x + y0] + sub_world[x2 + y0]                + sub_world[x0 + y1] + sub_world[x2 + y1]
                + sub_world[x0 + y2] + sub_world[x + y2] + sub_world[x2 + y2];
                // rule B36/S23
                sub_results[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !sub_world[x + y1])
                || (aliveCells == 2 && sub_world[x + y1]) ? 1 : 0;
            }
        }
        request_index = 0;
        HL_swap(&sub_world,&sub_results);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == 0) {
         end_time = MPI_Wtime();
         printf("Start_Time = %f\n",start_time);
         printf("End_Time = %f\n",end_time);
    }
    
    // Use MPI_Gather to assemble the Cell Universe to Rank 0
    // for small test worlds only -- 64x64
    // Output world using serial function provided in template
    MPI_Gather(sub_world+worldSize,cells_per_rank,MPI_UNSIGNED_CHAR,g_data,cells_per_rank,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

    // print final output
    bool print_arg = false;
    if (myrank == 0 && print_arg) {HL_printWorld(iterations);}

    MPI_Finalize();

    free(g_data);
    free(g_resultData);
    free(sub_world);
    free(sub_results);
    free(send_top_row);
    free(send_bottom_row);
    
    return true;
}