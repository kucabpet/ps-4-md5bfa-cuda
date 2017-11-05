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
#include <stdint.h>

#define COUNT_LETTER 2
#define COUNT_UNIT8_T_HASH 16

// Declaration
void hash_md5(char *input, uint8_t *result);
void show_hash(uint8_t *input);

// Demo kernel for array elements multiplication.
// Every thread selects one element and multiply it.

__global__ void kernel_mult( char *words, const int height, const int width)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	
	// if grid is greater then length of array...
	if ( l >= height) {
		return;
	}

	char word[COUNT_LETTER+1];
	for(int j=0; j < width; j++) {
		word[j] = words[width * l + j];
	}
	word[width] = '\0';


	uint8_t current_hash[COUNT_UNIT8_T_HASH];
	hash_md5(word, current_hash);

	printf("debug: index=%d, word=%s \n", l, word);
/*
	printf("hash=");
	show_hash(current_hash);

	printf("\n");
*/
	//pole[ l ] *= Mult;
}



void run_mult(char *words, const int height, const int width)
{
/*
	printf("xxxxxxxxxxxxxxxxxx");
	cudaError_t cerr;
	thread_hierarchy<<< dim3( 2, 2 ), dim3( 3, 3 )>>>();
	
	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
			printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	cudaDeviceSynchronize();
*/	


	cudaError_t cerr;
	int threads = 128;
	int length = height * width;
	int blocks = ( length + threads - 1 ) / threads;

	// Memory allocation in GPU device
	char *cWords;
	cerr = cudaMalloc( &cWords, length * sizeof( char ) );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from PC to GPU device
	cerr = cudaMemcpy( cWords, words, length * sizeof( char ), cudaMemcpyHostToDevice );
	if ( cerr != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Grid creation
	kernel_mult<<< blocks, threads >>>( cWords, height, width );

	if ( ( cerr = cudaGetLastError() ) != cudaSuccess )
		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Copy data from GPU device to PC
//	cerr = cudaMemcpy( P, cudaP, Length * sizeof( float ), cudaMemcpyDeviceToHost );
//	if ( cerr != cudaSuccess )
//		printf( "CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString( cerr ) );

	// Free memory
	cudaFree( cWords );

}
