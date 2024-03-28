//Johns Assignment 4 highlife_mpi.c
//Copied and modified from ass04/scratch-shared/highlife_mpi.c

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

static inline void HL_initReplicator( unsigned char **currGrid, size_t worldWidth, size_t worldHeight )
{
    size_t x, y;

    x = worldWidth/2;
    y = worldHeight/2;

    (*currGrid)[x + y*worldWidth + 1] = 1;
    (*currGrid)[x + y*worldWidth + 2] = 1;
    (*currGrid)[x + y*worldWidth + 3] = 1;
    (*currGrid)[x + (y+1)*worldWidth] = 1;
    (*currGrid)[x + (y+2)*worldWidth] = 1;
    (*currGrid)[x + (y+3)*worldWidth] = 1;
}

static inline void HL_swap( unsigned char **pA, unsigned char **pB)
{
  unsigned char *temp = *pA;
  *pA = *pB;
  *pB = temp;
}


//!!!!!!Might need to change order to reflect alex's method!!!!!!!!!
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
    MPI_Isend(currGrid + worldWidth, worldWidth, MPI_UNSIGNED_CHAR, prevRank, 0, MPI_COMM_WORLD, &sendRequests[0]); //Sends bottom ghost row to top rank (my top actual)
    MPI_Isend(currGrid + (numRows - 2) * worldWidth, worldWidth, MPI_UNSIGNED_CHAR, nextRank, 1, MPI_COMM_WORLD, &sendRequests[1]); //sends top ghost row to bottom rank (my buttom actual)

    // Receive of the bottom ghost row from the next rank and top ghost row from the previous rank
    MPI_Irecv(currGrid + (numRows - 1) * worldWidth, worldWidth, MPI_UNSIGNED_CHAR, nextRank, 0, MPI_COMM_WORLD, &recvRequests[0]); //Replace my bottom ghost from buttom rank
    MPI_Irecv(currGrid, worldWidth, MPI_UNSIGNED_CHAR, prevRank, 1, MPI_COMM_WORLD, &recvRequests[1]); //Replace my top ghost from top rank

    // Wait for all operations to complete before we continue.
    // This makes sure that each rank will have the necessary data before updating their cells
    MPI_Waitall(2, sendRequests, MPI_STATUSES_IGNORE);
    MPI_Waitall(2, recvRequests, MPI_STATUSES_IGNORE);
}


int main(int argc, char *argv[])
{
    //Setup MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //Start time with MPI_Wtime.
    double startTime;
    double endTime;
    if (rank == 0) {
        startTime = MPI_Wtime();
    }
    
    //Basic argument handling
    //Checking the arguments
    if (argc < 5) {
        if (rank == 0) {
            printf("Highlife requires at least 4 arguments: %s 1: pattern, 2: worldSize, 3: iterations, 4: thread count, 5: output (optional) \n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
    }
    //Grab arguments
    int pattern, iterations, threadCount, worldSize;
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

    //Allocate My Rank’s chunk of the universe, init pattern 5 (middle of my rank’s chuallocate space for "ghost" rows.
    // Initialize variables
    size_t realRows = worldSize;  //how many rows each rank will handle
    size_t totalNumRows = realRows + 2;  //total rows in a +2"grid" including ghost rows
    size_t dataLength = totalNumRows * worldSize; //total number of cells in the +2"grid"
    //These will store the current ranks calculations
    unsigned char *currGrid = (unsigned char *)calloc(dataLength,sizeof(unsigned char)); //+2grid
    unsigned char *singleGrid = (unsigned char *)calloc(worldSize*worldSize,sizeof(unsigned char)); //needed data +0grid
    unsigned char *nextGrid = (unsigned char *)calloc(dataLength,sizeof(unsigned char)); //next +2grid
    HL_initReplicator(&singleGrid, worldSize, worldSize);
    memcpy(currGrid + worldSize, singleGrid, worldSize*worldSize); //

    // Parallel iteration
    for (int i = 0; i < iterations; i++) {
        // Swap ghost rows
        //Exchange row data with MPI Ranks using MPI_Isend/Irecv.
        //Use MPI_Wait or MPI_Waitall to ensure all message are sent/recv’ed.
        swapGhostRows(currGrid, rank, size, worldSize, totalNumRows);
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank==0)
            printf("Iteration = %d\n", i);

        // Do rest of universe update as done in assignment 2 using CUDA HighLife kernel.
        HL_kernelLaunch(&currGrid, &nextGrid, worldSize, totalNumRows, threadCount, rank);

        // Swap grids for the next iteration
        HL_swap(&currGrid, &nextGrid);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //if Rank 0, end time with MPI_Wtime and printf MPI_Wtime performance results;
    if (rank == 0) {
        endTime = MPI_Wtime();
        printf("Performance time: %f seconds\n", endTime - startTime);
    }

    //if (Output Argument is True) { Printf my Rank’s chunk of universe. }
    if(true) { // use the 'output' variable to control this block
        int i, j;
        printf("Print World - Iteration: %d Rank: %d\n", iterations, rank);
        for(i = 0; i < worldSize; i++) {
            printf("Row %2d: ", i);
            for(j = 0; j < worldSize; j++) {
                printf("%u ", (unsigned int)currGrid[((i+1)*worldSize) + j]);
            }
            printf("\n");
        }
        printf("\n\n");
    }

    //Free variables and finalize out MPI
    free(currGrid);
    free(singleGrid);
    free(nextGrid);
    MPI_Finalize();

    return 0;
}