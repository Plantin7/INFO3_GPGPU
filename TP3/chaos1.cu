#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "cuStopwatch.cu"

// Compute sum of integers from 0 to n-1
__global__ void trianglenumber(uint64_t* res, uint64_t n) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < n) {
        atomicAdd(res, tid);         // *res += tid;
    }
	return;
}

int main() {
    // Allocate memory
    uint64_t *res_host, *res_dev;
    cudaHostAlloc((void**)&res_host, sizeof(uint64_t), cudaHostAllocDefault);
    cudaMalloc((void**)&res_dev, sizeof(uint64_t));

	// Perform computation
    cuStopwatch sw1;
    sw1.start();
	trianglenumber<<<1024, 1024>>>(res_dev, 1024*1024);
    cudaMemcpyAsync(res_host, res_dev, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Computation time: %.4fms\n", sw1.stop());
    printf("Result: %I64u\n", *res_host);
    
    // Free memory
    cudaFree(res_dev);
    cudaFreeHost(res_host);
	return 0;
}

/*
	Le problème ici c'est que nous avions un problème de concurrence lorsque nous calculions la somme de tous les entiers du tableau res.
	Etant donné que l'opération += n'est pas atomique alors les valeurs lu et écrite par les différents threads sont fauses (data race *res)
	=> 1024 * 1024 threads pbl de concurrence, accés de res !

*/