// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage
// Multiplication of elements in float array
//
// ***********************************************************************

#include <cuda_runtime.h>
#include <stdio.h>

// Demo kernel for array elements multiplication.
// Every thread selects one element and multiply it.
__global__ void kernel_mult( float *pole, int L, float Mult )
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	// if grid is greater then length of array...
	if ( l >= L ) return;

	pole[ l ] *= Mult;
}

// Demo kernel will display all global variables of grid organization.
// Warning! Function printf is available from compute capability 2.x
__global__ void thread_hierarchy()
{
    // Global variables
    // Grid dimension -				gridDim
	// Block position in grid -		blockIdx
	// Block dimension -			blockDim
	// Thread position in block -	threadIdx
    printf( "Block{%d,%d}[%d,%d] Thread{%d,%d}[%d,%d]\n",
	    gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
		blockDim.x, blockDim.y, threadIdx.x, threadIdx.y );
}

void run_mult(char **words, int dim, int len)
{
	cudaError_t cerr;
	int threads = 128;
	int blocks = ( Length + threads - 1 ) / threads;

	// Memory allocation in GPU device
	char **cudaP;
	cerr = cudaMalloc( &cudaP, Length * sizeof( float ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from PC to GPU device
	cerr = cudaMemcpy( cudaP, P, Length * sizeof( float ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Grid creation
	kernel_mult<<< blocks, threads >>>( cudaP, Length, Mult );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
	cerr = cudaMemcpy( P, cudaP, Length * sizeof( float ), cudaMemcpyDeviceToHost );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Free memory
	cudaFree( cudaP );
}
