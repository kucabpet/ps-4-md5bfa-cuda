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
/*
__global__ void kernel_mult( float *pole, int L, float Mult )
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	// if grid is greater then length of array...
	if ( l >= L ) return;

	pole[ l ] *= Mult;
}
*/

// Demo kernel will display all global variables of grid organization.
// Warning! Function printf is available from compute capability 2.x
__global__ void thread_hierarchy()
{
    // Global variables
    // Grid dimension -				gridDim
	// Block position in grid -		blockIdx
	// Block dimension -			blockDim
	// Thread position in block -	threadIdx
    printf( "Block{%d,%d}[%d,%d] Thread{%d,%d}[%d,%d] - something: %d \n",
	    gridDim.x, gridDim.y, blockIdx.x, blockIdx.y,
		blockDim.x, blockDim.y, threadIdx.x, threadIdx.y,
		blockDim.x * blockIdx.x + threadIdx.x);
}

void run_mult(char *words, int height, int width)
{

	cudaError_t cerr;
	thread_hierarchy<<< dim3( 2, 2 ), dim3( 3, 3 )>>>();
	
	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
			printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	cudaDeviceSynchronize();
	

/*
	cudaError_t cerr;
	int threads = 128;
	int blocks = ( height * width + threads - 1 ) / threads;

	// Memory allocation in GPU device
//	char *cWords;
//	cerr = cudaMalloc( &cWords, height * width * sizeof( char ) );
//	if ( cerr != cudaSuccess )
//		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from PC to GPU device
//	cerr = cudaMemcpy( cudaP, P, Length * sizeof( float ), cudaMemcpyHostToDevice );
//	if ( cerr != cudaSuccess )
//		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Grid creation
//	kernel_mult<<< blocks, threads >>>( cudaP, Length, Mult );
	thread_hirearchy<<< blocks, threads >>>();

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
//	cerr = cudaMemcpy( P, cudaP, Length * sizeof( float ), cudaMemcpyDeviceToHost );
//	if ( cerr != cudaSuccess )
//		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Free memory
//	cudaFree( cudaP );
*/
}
