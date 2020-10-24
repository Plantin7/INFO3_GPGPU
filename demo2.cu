#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void kernel() {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t n = tid;
	uint32_t sum = 0;
    uint32_t prod = 1;
    while(n != 0){
        uint32_t digit = n % 10;
        n /= 10;
        sum += digit;
        prod *= digit;
    }
    if(sum*prod == tid) printf("%u\n", tid);
	return;
}

void checkrange(uint32_t range){
    double dim = sqrt(range);                                      // 1024 : 32 / 16777216 : 4096
    uint32_t nbThreads = (uint32_t)ceil(range/(dim));
    if (nbThreads >= 1024){
        nbThreads = 1024;
    }
    printf("Checking %u for sum-product numbers\n", range);
    kernel<<<(uint32_t)dim, nbThreads, 0 >>>();  // 1024 : 32 blocks ; 32 threads / 16777216 : 4096 blocks ;  4096 Threads
    cudaDeviceSynchronize();
}

int main() {
	// main iteration
	checkrange(1024);
    checkrange(16777216);
	return 0;
}

/*
ceil : arrondi au supérieur
There is a limit to the number of threads per block, since all threads of a block are expected to reside on the same processor core and must share the limited memory resources of that core. 
On current GPUs, a thread block may contain up to 1024 threads. 

*/