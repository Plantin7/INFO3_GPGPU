#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int devices;
    cudaDeviceProp prop;
    
    try {
        cudaGetDeviceCount(&devices);
        for(int device = 0; device < devices; device++) {
        	cudaGetDeviceProperties(&prop, device);
        	std::cout << "Device Number                : " << device << std::endl;
        	std::cout << "Device name                  : " << prop.name << std::endl;
        	std::cout << "Memory Clock Rate (KHz)      : " << prop.memoryClockRate << std::endl;
        	std::cout << "Global Memory size (bits)    : " << prop.memoryBusWidth << std::endl;
        	std::cout << "Warp Size                    : " << prop.warpSize << std::endl;
        	std::cout << "Peak Memory Bandwidth (GB/s) : " << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl;
        }
    }
    catch (const cudaError_t & e) {
    	std::cerr << e;
    }

	return 0;
}