//highlife_mpi.c

#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include<string.h>
#include<mpi.h>

extern bool HL_kernelLaunch(unsigned char **d_data,
                         unsigned char **d_resultData,
                         size_t worldWidth,
                         size_t worldHeight,
                         int threadsCount,
                         int rank);

static inline void HL_initReplicator( unsigned char **curGrid, size_t worldWidth, size_t worldHeight )
{
    size_t x, y;

    x = worldWidth/2;
    y = worldHeight/2;

    (*curGrid)[x + y*worldWidth + 1] = 1;
    (*curGrid)[x + y*worldWidth + 2] = 1;
    (*curGrid)[x + y*worldWidth + 3] = 1;
    (*curGrid)[x + (y+1)*worldWidth] = 1;
    (*curGrid)[x + (y+2)*worldWidth] = 1;
    (*curGrid)[x + (y+3)*worldWidth] = 1;
}

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
  unsigned char *temp = *pA;
  *pA = *pB;
  *pB = temp;
}

// Don't Modify this function or your submitty autograding will not work
static inline void HL_printWorld(unsigned char **gatherWorld, size_t worldWidth, size_t worldHeight, size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);

    for( i = 0; i < worldHeight; i++)
    {
        printf("Row %2d: ", i);
        for( j = 0; j < worldWidth; j++)
        {
            printf("%u ", (unsigned int)(*gatherWorld)[(i*worldWidth) + j]);
        }
        printf("\n");
    }

    printf("\n\n");
}

// This function is used to swap ghost rows so that each local grid will have enough context to
// update their particular cell. The top ghost row is filled with the bottom edge row from the process directly above it,
// and the bottom ghost row is filled with the top edge row from the process directly below it.
void swapGhostRows(unsigned char* currGrid, int rank, int size, size_t worldWidth, size_t numRows) {

    // Get the previous or next rank, including the logic for wrap around.
    int prevRank = (rank - 1 + size) % size;
    int nextRank = (rank + 1) % size;

    MPI_Request sendRequests[2], recvRequests[2];

    //Initialization of the sends
    // Sends the top real row to the previous rank, and sends the bottom real row to the next rank
    MPI_Isend(currGrid + worldWidth, worldWidth, MPI_UNSIGNED_CHAR, prevRank, 0, MPI_COMM_WORLD, &sendRequests[0]);
    MPI_Isend(currGrid + (numRows - 2) * worldWidth, worldWidth, MPI_UNSIGNED_CHAR, nextRank, 1, MPI_COMM_WORLD, &sendRequests[1]);

    // Receive of the bottom ghost row from the next rank and top ghost row from the previous rank
    MPI_Irecv(currGrid + (numRows - 1) * worldWidth, worldWidth, MPI_UNSIGNED_CHAR, nextRank, 0, MPI_COMM_WORLD, &recvRequests[0]);
    MPI_Irecv(currGrid, worldWidth, MPI_UNSIGNED_CHAR, prevRank, 1, MPI_COMM_WORLD, &recvRequests[1]);

    // Wait for all operations to complete before we continue.
    // This makes sure that each rank will have the necessary data before updating their cells
    MPI_Waitall(2, sendRequests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2, recvRequests, MPI_STATUSES_IGNORE);
}


int main(int argc, char *argv[])
{
    //Setup MPI
    MPI_Init(&argc, &argv);
    int rank, size, pattern, iterations, threadCount;
    unsigned long worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Checking the arguments
    if (argc < 5) {
        if (rank == 0) {
            printf("Highlife requires at least 4 arguments: %s 1: pattern, 2: worldSize, 3: iterations, 4: thread count, 5: output (optional) \n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
    }

    //Grab arguments
    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    threadCount = atoi(argv[4]);
    
    //For checking correctness
    bool output;
    if(argc > 4){
        output = strcmp(argv[5], "true") == 0;
    }else{
        output = false;
    }

    // Initialize variables
    size_t realRows = worldSize;  //how many rows each rank will handle
    size_t totalNumRows = realRows + 2;  //total rows in a "grid" including ghost rows
    size_t dataLength = totalNumRows * worldSize; //total number of cells in the "grid"

    // if(rank==0)
    //     printf("%d\n", totalNumRows);

    //Initialize for the time
    double startTime;
    double endTime;

    //These will store the current ranks calculations
    unsigned char *currGrid = (unsigned char *)calloc(dataLength,sizeof(unsigned char));
    unsigned char *singleGrid = (unsigned char *)calloc(worldSize*worldSize,sizeof(unsigned char));
    unsigned char *nextGrid = (unsigned char *)calloc(dataLength,sizeof(unsigned char));

    // Initialization
    if (rank == 0) {
        startTime = MPI_Wtime();
    }

    HL_initReplicator(&singleGrid, worldSize, worldSize);    // Shouldnt this be &currGrid,,worldSize/# ranks
    memcpy(currGrid + worldSize, singleGrid, worldSize*worldSize); //!!!

    // Parallel iteration
    for (int i = 0; i < iterations; i++) {
        // Swap ghost rows
        swapGhostRows(currGrid, rank, size, worldSize, totalNumRows); //!!!
        if(rank==0)
            printf("Iteration = %d\n", i);
        HL_kernelLaunch(&currGrid, &nextGrid, worldSize, worldSize+2, threadCount, rank); //!!!

        // Swap grids for the next iteration
        unsigned char *temp = currGrid;
        currGrid = nextGrid;
        nextGrid = temp;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        endTime = MPI_Wtime();
        printf("Performance time: %f seconds\n", endTime - startTime);
    }

    // Gather the computed slices back to the root process
    // If user enters an output argument

        //allocate memory to get the final world.
        unsigned char *gatherWorld = NULL;
        if (rank == 0) {
            gatherWorld = (unsigned char *)calloc(worldSize * worldSize * size,sizeof(unsigned char));
        }

        MPI_Gather(currGrid + worldSize, realRows * worldSize, MPI_UNSIGNED_CHAR, gatherWorld, realRows * worldSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    if (output) {
        //copy the gathered world into g_data to print
        if (rank == 0) {
            HL_printWorld(&gatherWorld, worldSize, worldSize*size, iterations);
        }
    }

    //Free variables and finalize out MPI
    free(currGrid);
    free(singleGrid);
    free(nextGrid);
    free(gatherWorld);
    MPI_Finalize();

    return 0;
}