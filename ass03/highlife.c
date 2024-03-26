#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

// Global variables are maintained but will be used differently in MPI context.
unsigned char *g_resultData = NULL;
unsigned char *g_data = NULL;
size_t g_worldWidth = 0;
size_t g_worldHeight = 0;
size_t g_dataLength = 0; // g_worldWidth * g_worldHeight

// Existing HighLife initializations
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

// MPI-specific variables
int myrank, numranks;
unsigned char *local_data = NULL;
unsigned char *local_resultData = NULL;
size_t local_height; // Height of the local chunk

// Modify HL_initMaster to only initialize the full grid in rank 0
// Other ranks will use this to initialize their local chunk later
void HL_initMaster(unsigned int pattern, size_t worldWidth, size_t worldHeight) {
    // Initializations for rank 0
    if (myrank == 0) {
        g_worldWidth = worldWidth;
        g_worldHeight = worldHeight;
        g_dataLength = worldWidth * worldHeight;
        g_data = calloc(g_dataLength, sizeof(unsigned char)); 
        switch(pattern) {
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
    // Define local_height and allocate memory for local_data and local_resultData
    local_height = worldHeight / numranks;
    size_t local_dataLength = local_height * worldWidth;
    local_data = calloc(local_dataLength, sizeof(unsigned char));
    local_resultData = calloc(local_dataLength, sizeof(unsigned char));
}

// Add modified versions of your HighLife functions here, to operate on local_data and local_resultData
static inline unsigned int HL_countAliveCells(unsigned char* local_data, 
                                              unsigned char* top_ghost_row, 
                                              unsigned char* bottom_ghost_row, 
                                              size_t x0, 
                                              size_t x1, 
                                              size_t x2, 
                                              size_t y0, 
                                              size_t y1, 
                                              size_t y2, 
                                              size_t local_width, 
                                              size_t local_height) 
{
    unsigned int count = 0;
    // Check if we are at the top edge of the local chunk
    if (y0 == 0 && top_ghost_row != NULL) {
        // Use the top ghost row for the top row of neighbors
        count += top_ghost_row[x0] + top_ghost_row[x1] + top_ghost_row[x2];
    } else {
        // Use local data for the top row of neighbors
        count += local_data[x0 + y0] + local_data[x1 + y0] + local_data[x2 + y0];
    }

    // Middle row (always local data)
    count += local_data[x0 + y1] + local_data[x2 + y1];

    // Check if we are at the bottom edge of the local chunk
    if (y2 == (local_height - 1) * local_width && bottom_ghost_row != NULL) {
        // Use the bottom ghost row for the bottom row of neighbors
        count += bottom_ghost_row[x0] + bottom_ghost_row[x1] + bottom_ghost_row[x2];
    } else {
        // Use local data for the bottom row of neighbors
        count += local_data[x0 + y2] + local_data[x1 + y2] + local_data[x2 + y2];
    }

    return count;
}


void HL_swapLocal() {
    unsigned char *temp = local_data;
    local_data = local_resultData;
    local_resultData = temp;
}

int main(int argc, char *argv[]) {
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);

    // Argument parsing and initial setup
    if(argc != 4) {
        if(myrank == 0) {
            printf("HighLife requires 3 arguments: pattern number, size of the world, and number of iterations\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);

    // Initialize master or local universe depending on the rank
    HL_initMaster(pattern, worldSize, worldSize); //Already only accepts rank=0

    // Scatter the initial world to all processes
    MPI_Scatter(g_data, worldSize * local_height, MPI_UNSIGNED_CHAR,
                local_data, worldSize * local_height, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    
    
    // Size of one row
    size_t row_size = g_worldWidth * sizeof(unsigned char);
    // Allocate memory for ghost rows
    unsigned char *top_ghost_row = malloc(row_size);
    unsigned char *bottom_ghost_row = malloc(row_size);
    int above, below;
    // Main simulation loop
    for (int i = 0; i < iterations; ++i) {
        MPI_Status status;
        MPI_Request send_request[2], recv_request[2];

        above = myrank - 1;
        below = myrank + 1;

        // Handle wrap-around
        if (above < 0) above = numranks - 1;
        if (below == numranks) below = 0;

        // Send top row to the above rank and bottom row to the below rank
        MPI_Isend(local_data, row_size, MPI_UNSIGNED_CHAR, above, 0, MPI_COMM_WORLD, &send_request[0]);
        MPI_Isend(&local_data[row_size * (local_height - 1)], row_size, MPI_UNSIGNED_CHAR, below, 1, MPI_COMM_WORLD, &send_request[1]);

        // Receive bottom row from above rank and top row from below rank
        MPI_Irecv(top_ghost_row, row_size, MPI_UNSIGNED_CHAR, above, 1, MPI_COMM_WORLD, &recv_request[0]);
        MPI_Irecv(bottom_ghost_row, row_size, MPI_UNSIGNED_CHAR, below, 0, MPI_COMM_WORLD, &recv_request[1]);

        // Wait for all non-blocking sends and receives to complete
        MPI_Waitall(2, recv_request, MPI_STATUSES_IGNORE);
        MPI_Waitall(2, send_request, MPI_STATUSES_IGNORE);

        // Now, you need to integrate the received ghost rows into your computation
        // Note: Make sure you consider these rows during your cell updates but do not overwrite
        //       your actual data with these ghost rows.

        // Perform updates on local data
        for (size_t y = 0; y < local_height; ++y) {
            size_t y0, y1, y2; // Indices for previous, current, and next row

            // Calculate row indices considering ghost rows
            y1 = y * g_worldWidth; // Current row in local data
            y0 = (y == 0) ? 0 : (y - 1) * g_worldWidth; // Use top ghost row if y == 0
            y2 = (y == local_height - 1) ? 0 : (y + 1) * g_worldWidth; // Use bottom ghost row if y == local_height - 1

            for (size_t x = 0; x < g_worldWidth; ++x) {
                size_t x0 = (x + g_worldWidth - 1) % g_worldWidth;
                size_t x1 = x;
                size_t x2 = (x + 1) % g_worldWidth;

                // Calculate alive neighbors. Use ghost rows if y is at top or bottom
                unsigned int aliveCells = 0;
                if (y == 0) { // Top edge of local chunk
                    // Pass the top ghost row for the top neighbors
                    aliveCells = HL_countAliveCells(local_data, top_ghost_row, NULL, x0, x1, x2, 0, y1, y * g_worldWidth, g_worldWidth, local_height);
                } else if (y == local_height - 1) { // Bottom edge of local chunk
                    // Pass the bottom ghost row for the bottom neighbors
                    aliveCells = HL_countAliveCells(local_data, NULL, bottom_ghost_row, x0, x1, x2, (y - 1) * g_worldWidth, y1, 0, g_worldWidth, local_height);
                } else { // Interior cells
                    // Normal call with local data for both top and bottom neighbors
                    aliveCells = HL_countAliveCells(local_data, NULL, NULL, x0, x1, x2, (y - 1) * g_worldWidth, y1, (y + 1) * g_worldWidth, g_worldWidth, local_height);
                }

                // Apply the HighLife rules to compute the next state
                local_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !local_data[x + y1])
                                        || (aliveCells == 2 && local_data[x + y1]) ? 1 : 0;
            }
        }

       

        // Swap pointers between local_data and local_resultData if necessary
        HL_swapLocal();

        // Ensure all ranks completed the iteration
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // Free ghost row memory after use
    free(top_ghost_row);
    free(bottom_ghost_row);
    // Gather the local matrices back to the master rank
    if(myrank == 0) {
        // Prepare buffer to gather the full matrix
        // Use MPI_Gather to collect all local matrices

        // Only if you need to gather the whole world back to rank 0
        if (false) {
            MPI_Gather(local_data, local_height * g_worldWidth, MPI_UNSIGNED_CHAR, 
                    g_data, local_height * g_worldWidth, MPI_UNSIGNED_CHAR, 
                    0, MPI_COMM_WORLD);
        }
    }

    // Finalize MPI
    MPI_Finalize();

    if(myrank == 0) {
        // Free global memory and possibly print the final world
        free(g_data);
        free(g_resultData);
    }

    // Free local memory
    free(local_data);
    free(local_resultData);

    return 0;
}
