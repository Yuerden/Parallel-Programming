//  barn-shared>assignment3>:

#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

// Result from last compute of world.
unsigned char *g_resultData=NULL;

// Current state of world.
unsigned char *g_data=NULL;

size_t g_worldWidth=0; 
/// Current height of world.
size_t g_worldHeight=0;

/// Current data length (product of width and height)
size_t g_dataLength=0;  // g_worldWidth * g_worldHeight
// "Above" row
unsigned char *g_aboveRow=NULL; 

// "Below" row
unsigned char *g_belowRow=NULL;

unsigned char *local_data=NULL;

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

/// Serial version of standard byte-per-cell life.
bool HL_iterateSerial(size_t iterations)
{
  size_t i, y, x;
  for (i = 0; i < iterations; ++i){
    for (y = 0; y < g_worldHeight; ++y) {
      size_t y0 = ((y + g_worldHeight - 1) % g_worldHeight) * g_worldWidth;
      size_t y1 = y * g_worldWidth;
      size_t y2 = ((y + 1) % g_worldHeight) * g_worldWidth;
      
      for (x = 0; x < g_worldWidth; ++x) {
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

int main(int argc, char *argv[]){
  unsigned int pattern = 0;
  unsigned int worldSize = 0;
  unsigned int iterations = 0;

  int rank, num_rank;
  
  pattern = atoi(argv[1]);
  worldSize = atoi(argv[2]);
  iterations = atoi(argv[3]);
  
  g_worldHeight = worldSize;
  g_worldWidth = worldSize;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_rank);
  
  MPI_Status stat;
  MPI_Request send_request, recv_request;

  int aboveRank = (rank+num_rank+1) % num_rank;
  int belowRank = (rank+num_rank-1) % num_rank;

  printf("num_rank: %d\n", num_rank);
  double t0, t1;
  if(rank == 0){
    t0 = MPI_Wtime();
    int total_world_size = worldSize * worldSize;
    // initialize data
    g_data = calloc(total_world_size, sizeof(unsigned char));
    // initialize resulting data
    g_resultData = calloc(total_world_size, sizeof(unsigned char));
    // initialize above row
    g_aboveRow = calloc(worldSize, sizeof(unsigned char));
    // initialize below row
    g_belowRow = calloc(worldSize, sizeof(unsigned char));
    
    int x, y;
    x = worldSize / 2;
    y = worldSize / 2;
    
    g_data[x + y*worldSize + 1] = 1;
    g_data[x + y*worldSize + 2] = 1;
    g_data[x + y*worldSize + 3] = 1;
    g_data[x + (y+1)*worldSize] = 1;
    g_data[x + (y+2)*worldSize] = 1;
    g_data[x + (y+3)*worldSize] = 1;

    unsigned char *recvbuf;
    recvbuf = calloc(total_world_size, sizeof(unsigned char));

    MPI_Scatter(g_data, total_world_size / num_rank, MPI_UNSIGNED_CHAR, recvbuf, total_world_size / num_rank, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
  }else{
    int local_size = worldSize * worldSize / num_rank;
    local_data = calloc(local_size, sizeof(unsigned char));
    MPI_Scatter(NULL, 0, MPI_DATATYPE_NULL, local_data, local_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD); 
  }

  MPI_Barrier( MPI_COMM_WORLD );

  int tag1 = 10, tag2 = 20;
  int tick;

  for(tick = 0; tick < iterations; ++tick){
    // send & receive for above row
    MPI_Irecv(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag1, MPI_COMM_WORLD, &recv_request);
    MPI_Isend(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag1, MPI_COMM_WORLD, &send_request);

    MPI_Wait(&send_request, &stat);
    MPI_Wait(&recv_request, &stat);

    // send & receive for below row
    MPI_Irecv(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag2, MPI_COMM_WORLD, &recv_request); 
    MPI_Isend(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag2, MPI_COMM_WORLD, &send_request);

    MPI_Wait(&send_request, &stat);
    MPI_Wait(&recv_request, &stat);

    // Swap
    HL_swap(&g_aboveRow, &g_belowRow);
    MPI_Barrier( MPI_COMM_WORLD );

    // Serial computation
    int x, y;
    for (y = 0; y < g_worldHeight; ++y) {
      size_t y0 = ((y + g_worldHeight - 1) % g_worldHeight) * g_worldWidth;
      size_t y1 = y * g_worldWidth;
      size_t y2 = ((y + 1) % g_worldHeight) * g_worldWidth;
      for (x = 0; x < g_worldWidth; ++x) {
        size_t x0 = (x + g_worldWidth - 1) % g_worldWidth;
        size_t x2 = (x + 1) % g_worldWidth;

        unsigned int aliveCells = HL_countAliveCells(g_data, x0, x, x2, y0, y1, y2);
        // rule B36/S23
        g_resultData[x + y1] = (aliveCells == 3) || (aliveCells == 6 && !g_data[x + y1])    
        || (aliveCells == 2 && g_data[x + y1]) ? 1 : 0;
      }
    }
    HL_swap(&g_data, &g_resultData);
  }

  if(rank == 0){
    t1 = MPI_Wtime();
    printf("\n[WORLD SIZE] %d x %d\n[ITERATIONS] %d\n[EXECUTION TIME] %f\n", worldSize, worldSize, iterations, t1-t0);
  }

  MPI_Barrier( MPI_COMM_WORLD );
  MPI_Finalize();
  if(worldSize <= 64){
    int i, j;
    printf("Print World - Iteration %d \n", iterations);
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
  free(g_data);
  free(g_resultData);
}