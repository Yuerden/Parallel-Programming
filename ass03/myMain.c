//CURRENT TODOs:
    //FIX OUTPUT IF STATEMENT WITH FIXING MPI_GATHER CALLs' FOR ALL RANKS
int main(int argc, char *argv[]) {
    
    // Variables for the simulation
    int myrank, numranks;
    int rootrank = 0;
    int pattern, world_N, iterations;
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
    printf("Pattern: %d, World_N: %d, Iterations: %d\n", pattern, world_N, iterations);
    printf("Splitting now\n\n")

    // Setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    printf("Rank: %d is here! \t", myrank);

    
    // Calculate chunk stuff
    int chunkHeight = world_N / numranks;
    int chunkSize = world_N * chunkHeight;
    unsigned char *chunk_data=NULL;
    unsigned char *ghostAbove=NULL;
    unsigned char *ghostBellow=NULL;


    printf("Rank %d: has greated chunk_data of ChunkHeight: %d, WorldWidth (world_N): %d, with a total ChunkSize: %d\n", myrank, chunkHeight, world_N, chunckSize);
    
    // Initialization specific for Rank 0
    if (myrank == rootrank) {
        startTime = MPI_Wtime();

        // Initialize a local copy of the whole NxN Cell World to global g_data at top of program
        HL_initMaster(pattern, world_N, world_N);
        
        // Allocation "chunk" of Cell Universe plus "ghost" rows for RootRank
        chunk_data = calloc(chunkSize, sizeof(unsigned char));
        //INSERT GHOST HERE::::::

        printf("Rank 0: World initialized. Width: %zu, Height: %zu\n", g_worldWidth, g_worldHeight);

        // Scatter the initialized world to all ranks
        // Since Rank 0 also needs a part of the world, use MPI_Scatter with special handling
        unsigned char* scatterBuffer = malloc(world_N * world_N * sizeof(unsigned char)); // Temporary buffer for scattering
        memcpy(scatterBuffer, g_data, world_N * world_N * sizeof(unsigned char)); // Copy initialized world from global data pointer into buffer
        
        MPI_Scatter(scatterBuffer, chunkSize, MPI_UNSIGNED_CHAR,
                    chunk_data, chunkSize, MPI_UNSIGNED_CHAR, rootrank, MPI_COMM_WORLD);
        
        free(scatterBuffer);
    } else {
        // Allocation "slice" of universe plus "ghost" rows;
        chunk_data = calloc(chunkSize, sizeof(unsigned char));
        //INSERT GHOST HERE::::::

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
        unsigned int aliveCells
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
                elif(y==chunkHeight-1){
                    aliveCells = chunk_data[x0 + y0] + chunk_data[x + y0] + chunk_data[x2 + y0]
                               + chunk_data[x0 + y1]                       + chunk_data[x2 + y1]
                               + myBottomGhost[x0] + myBottomGhost[x] + myBottomGhost[x2];
                }else{
                    aliveCells = chunk_data[x0 + y0] + chunk_data[x + y0] + chunk_data[x2 + y0]
                               + chunk_data[x0 + y1]                       + chunk_data[x2 + y1]
                               + chunk_data[x0 + y2] + chunk_data[x + y2] + chunk_data[x2 + y2];
                }
                //else alive cells normal

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

    //OUTPUT:
        //NOTE THAT THE GATHER FUNCTION ACTS MUCH LIKE SCATTER AND THUS NEEDS IF-ROOT-ELSE GATHER CALLS
        //THIS ^ IMPLIES THAT THIS SECTION IS NON-FUNCTIONING IN ITS CURRENT STATE (hence print = false)
    bool print = false;
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
    MPI_Finalize();
    return 0;
}