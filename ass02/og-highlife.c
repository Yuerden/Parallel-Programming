#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

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
    unsigned int pattern = 0;
    unsigned int worldSize = 0;
    unsigned int iterations = 0;

    printf("This is the HighLife running in serial on a CPU.\n");

    if( argc != 4 )
    {
	printf("HighLife requires 3 arguments, 1st is pattern number, 2nd the sq size of the world and 3rd is the number of itterations, e.g. ./highlife 0 32 2 \n");
	exit(-1);
    }

    pattern = atoi(argv[1]);
    worldSize = atoi(argv[2]);
    iterations = atoi(argv[3]);
    
    HL_initMaster(pattern, worldSize, worldSize);
    printf("AFTER INIT IS............\n");
    HL_printWorld(0);
    
    HL_iterateSerial( iterations );
    
    printf("######################### FINAL WORLD IS ###############################\n");
    HL_printWorld(iterations);

    free(g_data);
    free(g_resultData);
    
    return true;
}
