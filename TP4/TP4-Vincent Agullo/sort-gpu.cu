#include "cuStopwatch.cu"
#include "randgen.cpp"
#include <cuda_runtime.h>
#include <iostream>
#include <bitset>
#include <algorithm>    /* sort */
#include <iomanip>      /* fixed, setprecision*/

// Understand the principe of radix sort : https://www.youtube.com/watch?v=dPwAA7j-8o4

/*
	Vidéo Reférence : https://www.youtube.com/watch?v=iS0S7F2U4-o
	If the least-significant bit is 1 we put 1 into the out_bits array and put 0 on the revered_bits array
	and compute the number of ones (least significant bit) from the input array and store in nb_ones
*/
__global__ void getBit(uint64_t* input, uint64_t* reverse_out_bits, uint64_t* out_bits, uint64_t* nb_ones, uint64_t mask) {
	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if((input[tid] & mask) == mask) {
        reverse_out_bits[tid] = 0;
        out_bits[tid] = 1;
	}
	else {
        reverse_out_bits[tid] = 1;
        out_bits[tid] = 0;
		atomicAdd(nb_ones, 1); // Count the number of ones 
	}
}

/*
	vidéo reference : https://www.youtube.com/watch?v=mmYv3Haj6uc  - Blelloch Scan !
	Source code inspiration : https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

	Compute the sum with the all previous elements and for each element in the output_data 
*/
__global__ void blelloch_scan(uint64_t *output_data, uint32_t n) { 
	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint64_t offset = 1; 

	for (int d = n>>1; d > 0; d >>= 1){ // build sum in place up the tree
		__syncthreads();    
		if (tid < d)    {
			uint64_t ai = offset*(2*tid+1)-1;     
			uint64_t bi = offset*(2*tid+2)-1;  
			output_data[bi] += output_data[ai];   
		}    
		offset *= 2; 
	}
	if (tid == 0) { 
		output_data[n - 1] = 0; // clear the last element   
	}
	//down
	for (int d = 1; d < n; d *= 2){   // traverse down tree & build scan 
		offset >>= 1;      
		__syncthreads();      
		if (tid < d) { 
			uint64_t ai = offset*(2*tid+1)-1;     
			uint64_t bi = offset*(2*tid+2)-1;
			float t = output_data[ai];
			output_data[ai] = output_data[bi]; 
			output_data[bi] += t;       
		}  
	}               
}


/*
	Vidéo Reférence : https://www.youtube.com/watch?v=iS0S7F2U4-o
	rearrange the array so as to put the correct values ​​at the position indicated by the blelloch scan algorithm
	if the bits is 1 it significate that we move the value from input array (its least significant bit is 0) to the good position (getted from the result scan array)
	else we need to place the value (its least significant bit is 1), we need to shift the array the number of times it was counted by 1 to put the value in the correct position.
*/
__global__ void sort_array(uint64_t* input, uint64_t* output, uint64_t* bits, uint64_t* result_scan, uint64_t* second_scan, uint64_t* num_ones) {
	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(bits[tid] == 1)
	{
		uint64_t idx = result_scan[tid];   // get the position in the first scan 
		output[idx] = input[tid];     // put the value in the good position
	}
	else {
		uint64_t idx = second_scan[tid] + *num_ones; // value whose low weight bit is 1 therefore we shift a good number of 1 and we place from this index
		output[idx] = input[tid];               // put the value in the good position
	}
}

/*
	Call the invoke méthod to sort the arr_host with radix sort algorithm
*/
float sort_gpu(uint64_t* arr_host, uint32_t size){ // arr is input host
	
	cuStopwatch time;
	uint64_t* arr_device;			   // input device
	uint64_t* output_device;	       // output device
	uint64_t* bits_device;		       // bits device
	uint64_t* reverse_bits_device;     // bits reverse from bits_device
	uint64_t* nb_ones_device; 		   // number of ones device
	uint64_t* res_scan_device;		   // result scan device
	uint64_t* res_second_scan_device;  // result second scan device

	cudaMalloc(&arr_device,  			size*sizeof(uint64_t));
	cudaMalloc(&output_device, 			size*sizeof(uint64_t));
	cudaMalloc(&bits_device,   		    size*sizeof(uint64_t));
	cudaMalloc(&nb_ones_device,       		 sizeof(uint64_t));
	cudaMalloc(&res_scan_device, 	   	size*sizeof(uint64_t));
	cudaMalloc(&reverse_bits_device,  	size*sizeof(uint64_t));
	cudaMalloc(&res_second_scan_device, size*sizeof(uint64_t));
	
	// put into device, the input array
	cudaMemcpy(arr_device, arr_host, size*sizeof(uint64_t), cudaMemcpyHostToDevice);

	time.start(); 
	int nb_thread = size > (1<<10) ? (1<<10) : size;    // number of max thread per grid (1024) 
    int nb_block = ceil(size/nb_thread);                // calculate the right number of blocks to have 
	
	uint64_t mask = 1;
	for(int x = 0; x < 64; x++){
		cudaMemset(nb_ones_device, 0, sizeof(uint64_t)); // set 0 on the gpu for the numbers of ones
	
		// get two array : one with the latest bit of each value and another with the reverse bit
		getBit<<<nb_block, nb_thread>>>(arr_device, reverse_bits_device, bits_device, nb_ones_device, mask); // Kernel Invoke

		// First Scan on the fist array of bit
		cudaMemcpy(res_scan_device, reverse_bits_device, size*sizeof(uint64_t), cudaMemcpyDeviceToDevice);
		blelloch_scan<<<nb_block, nb_thread>>>(res_scan_device, size); // invoke blelloch scan on the array of bit
		
		// Second Scan on the second array of bit
		cudaMemcpy(res_second_scan_device, bits_device, size*sizeof(uint64_t), cudaMemcpyDeviceToDevice);
		blelloch_scan<<<nb_block, nb_thread>>>(res_second_scan_device, size); // invoke blelloch scan on the array of reversed bit

		sort_array<<<nb_block, nb_thread>>>(arr_device, output_device, reverse_bits_device, res_scan_device, res_second_scan_device, nb_ones_device);
		mask <<= 1; // change bit !

		cudaMemcpy(arr_device, output_device, size*sizeof(uint64_t),   cudaMemcpyDeviceToDevice);
	}

	// The sort is finished we can put the result into the arr
	cudaMemcpy(arr_host, arr_device, size*sizeof(uint64_t), cudaMemcpyDeviceToHost);

	// Free the cuda allocation
	cudaFree(reverse_bits_device);
	cudaFree(bits_device);
	cudaFree(res_scan_device);
	cudaFree(res_second_scan_device);
	cudaFree(nb_ones_device);
	cudaFree(output_device);
	cudaFree(arr_device);

	return time.stop()/1000;
}

// Used for debug
void displayArrayToBits(uint64_t* arr, size_t size) {
	std::cout << "[DEBUG] - Array printed in bits" << std::endl << std::endl;

	for(int i = 0; i < size; i++) {
		uint64_t n = arr[i];
		for(int j = 56; j > 0; j -= 8){
			std::cout << std::bitset<8>(n >> j) << " ";
		}
		std::cout << std::bitset<8>(n) << " <====> " << n << std::endl;
	}
	std::cout << std::endl;
}

/*
	Sort the array on the CPU
*/
float sort_cpu(uint64_t* arr, size_t size) {
	clock_t time = clock();                  				// start clock
	std::sort(arr, arr + size);			     			    // sort the array
	return ((double)(clock() - time)) / CLOCKS_PER_SEC ;	// stop clock
}

// MAIN
int main(int argc, char const* argv[])
{
	// Declaration of needed variables and arrays
	uint64_t nbIntegers;
	uint64_t* array_cpu;
	uint64_t* array_gpu;
	float elapsed;

	if(argc != 2){
		std::cerr << "Missing one parameter" << std::endl;
        return EXIT_FAILURE;
    }

    if(atoi(argv[1])){
    	int n = atoi(argv[1]);
    	if(n < 0 || n > 32) {
    		std::cerr << "The parameter must be greater than 0 and less than 32" << std::endl;
    		return EXIT_FAILURE;
    	}
    	nbIntegers = n;
    }
    else {
    	std::cerr << "[The parameter is not a number" << std::endl;
    	return EXIT_FAILURE;
    }

    uint64_t powNbIntegers = (1 << nbIntegers);
    std::cout << "[DEBUG] - Entered Number : "   << nbIntegers    << std::endl;
    std::cout << "[DEBUG] - Producing array of " << powNbIntegers << " ununsigned 64bit integers " << std::endl;
    std::cout << "[DEBUG] - Generating Random Array ..."          << std::endl;

    cudaMallocHost((void**)&array_cpu, powNbIntegers*sizeof(uint64_t));
    cudaMallocHost((void**)&array_gpu, powNbIntegers*sizeof(uint64_t));
    
    randgen(array_cpu, powNbIntegers);
    std::cout << "[DEBUG] - Generating Random Array...DONE\n"     << std::endl;

    cudaMemcpy(array_gpu, array_cpu, sizeof(uint64_t)*powNbIntegers, cudaMemcpyHostToHost);
    //displayArrayToBits(array_cpu, powNbIntegers); // DEBUG
	
	std::cout << "[DEBUG] - CPU version : Sorting Array" << std::endl;
	elapsed = sort_cpu(array_cpu, powNbIntegers);
	//displayArrayToBits(array_cpu, powNbIntegers); // DEBUG
	std::cout << "CPU version, runtime " << std::fixed << std::setprecision(4) << elapsed << " s\n" <<std::endl;

	std::cout << "[DEBUG] - GPU version : Sorting Array" << std::endl;
	elapsed = sort_gpu(array_gpu, powNbIntegers);
    std::cout << "GPU version, runtime " << std::fixed << std::setprecision(4) << elapsed << " s\n" <<std::endl;
    //displayArrayToBits(array_gpu, powNbIntegers); // DEBUG 

	std::cout << "END" << std::endl;
    cudaFree(array_cpu);
    cudaFree(array_gpu);
	return 0;
}



// int* input_units;
	// int* res_units_host;

	// int* input_units_device;
	// int* output_units; 
	// int* value_units;
	// cudaMallocHost((void**)&input_units, 6*sizeof(int));
	// cudaMallocHost((void**)&res_units_host, 6*sizeof(int));

	// cudaMalloc(&input_units_device, 6*sizeof(int));
	// cudaMalloc(&output_units, 6*sizeof(int));
	// cudaMalloc(&value_units, sizeof(int));

	// // Initialisation
	// input_units[0] = 2;
	// input_units[1] = 3;
	// input_units[2] = 4;
	// input_units[3] = 6;
	// input_units[4] = 7;
	// input_units[5] = 1;

	// for(int i = 0; i < 6; i++) {
	// 	int n = input_units[i];
	// 	std::cout << std::bitset<4>(n) << std::endl;
	// }

	// cudaMemcpy(input_units_device, input_units ,sizeof(int)*6, cudaMemcpyHostToDevice);
	// cudaMemset(value_units, 0, sizeof(int)); // set 0 on the gpu

	// getBit<<<1, 6>>>(input_units_device, output_units, value_units);

	// cudaMemcpy(res_units_host, output_units,sizeof(int)*6, cudaMemcpyDeviceToHost);

	// for(int i = 0; i < 6; i++) {
	// 	std::cout << input_units[i] << " : " << res_units_host[i] << std::endl;
	// }

	// uint64_t mask = 1;
	// for(int i = 0; i < 64; i++){
	// 	mask <<= 1;
	// 	std::cout << i << " : " << mask << std::endl; 
	// }

	// std::cout << "--------------- KERNEL TEST -------------- " << std::endl;

	// int* test_array_host;
	// int* test_array_device;
	// cudaMallocHost((void**)&test_array_host, 8*sizeof(int));
	// // cudaMalloc(&test_array_device, 8*sizeof(int));

	// // Initialisation - Array to short
	// test_array_host[0] = 23;
	// test_array_host[1] = 113;
	// test_array_host[2] = 34;
	// test_array_host[3] = 346;
	// test_array_host[4] = 7;
	// test_array_host[5] = 11;
	// test_array_host[6] = 34;
	// test_array_host[7] = 51;

	// // cudaMemcpy(test_array_device, test_array_host ,sizeof(int)*8, cudaMemcpyHostToDevice);
	// sort_kernel_test<<<1, 8>>>(test_array_host, 8);

	// std::cout << n << " <====> "<< std::bitset<8*sizeof(n)>(n) << std::endl; <- one shot
		/*std::cout << std::bitset<8>(n >> 56) << " "
				  << std::bitset<8>(n >> 48) << " "
				  << std::bitset<8>(n >> 40) << " "
			      << std::bitset<8>(n >> 32) << " "
			      << std::bitset<8>(n >> 24) << " "
				  << std::bitset<8>(n >> 16) << " "
				  << std::bitset<8>(n >> 8)  << " "
				  << std::bitset<8>(n)       << " <====> " << n << std::endl;*/

				  // ------------------------------------- ZONE DE TEST ------------------------------------------- //

	// int* test;
	// cudaMallocHost((void**)&test, 8*sizeof(int));
	// test[0] = 0;
	// test[1] = 1;
	// test[2] = 0;
	// test[3] = 1;
	// test[4] = 1;
	// test[5] = 1;
	// test[6] = 0;
	// test[7] = 0;

	// int* out;
	// cudaMallocHost((void**)&out, 8*sizeof(int));

	// int* result ;
	// cudaMalloc(&result, sizeof(int)*8);
	// cudaMemcpy(result, test ,sizeof(int)*8, cudaMemcpyHostToDevice);
	// blelloch_scan<<<1, 8>>>(result, 8); 

	// cudaMemcpy(out, result ,sizeof(int)*8, cudaMemcpyDeviceToHost);

	// for (int i = 0; i < 8; ++i)
	// {
	// 		std::cout << out[i]<< std::endl;
	// }

	// std::cout << "------------------------------------------------------" << std::endl;

	// cudaFree(test);
	// cudaFree(result);
	// cudaFree(out);
	
