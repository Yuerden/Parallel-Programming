all: highlife_mpi.c highlife.cu
	mpixlc g highlife_mpi.c -c -o highlife-mpi.o
	nvcc -g -G highlife.cu -c -o highlife-cuda.o
	mpicc -g highlife-mpi.o highlife-cuda.o -o highlife-exe \
		-L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
