#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel() {
	printf("%d, %d\n", threadIdx.x, blockIdx.x);
	return;
}

int main() {
	// main iteration
	kernel <<<16, 4, 0>>>();
	return 0;
}

/*
Exercice 2 : 

CUDA C++ extends C++ by allowing the programmer to define C++ functions, 
called kernels, that, when called, are executed N times in parallel by N different CUDA threads
Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through built-in variables. (threadIdx.x)

The number of threads per block and the number of blocks per grid specified in the <<<...>>> syntax can be of type int or dim3. 
Each block within the grid can be identified by a one-dimensional, two-dimensional, 
or three-dimensional unique index accessible within the kernel through the built-in blockIdx variable.

*/
// Va tourner sur le GPU
// 4 thread into 16 blocks