#include <cuda_runtime.h>
#include <iostream> 

__global__ void getBit(int* input, int* out_bits, int* reverse_out_bits, int* nbOnes, int mask) {
	int tid = threadIdx.x;

	if((input[tid] & mask) == mask) {
        out_bits[tid] = 0;
        reverse_out_bits[tid] = 1;
	}
	else {
        out_bits[tid] = 1;
        reverse_out_bits[tid] = 0;
		atomicAdd(nbOnes, 1); // Count the number of ones 
	}
}

// Blelloch Scan !
__global__ void blelloch_scan(int *output_data, int n) { 
	int tid = threadIdx.x;
	int offset = 1; 

	for (int d = n>>1; d > 0; d >>= 1){ // build sum in place up the tree
		__syncthreads();    
		if (tid < d)    {
			int ai = offset*(2*tid+1)-1;     
			int bi = offset*(2*tid+2)-1;  
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
			int ai = offset*(2*tid+1)-1;     
			int bi = offset*(2*tid+2)-1;
			float t = output_data[ai];
			output_data[ai] = output_data[bi]; 
			output_data[bi] += t;       
		}  
	}               
}

// rearrange the array so as to put the correct values ​​at the position indicated by the blelloch scan algorithm
__global__ void sort_array(int* input, int* output, int* bits, int* result_scan, int* second_scan, int* num_ones) {
	int tid = threadIdx.x;

	if(bits[tid] == 1)
	{
		int idx = result_scan[tid];   // get the position in the first scan 
		output[idx] = input[tid];     // put the value in the good position
	}
	else {
		int idx = second_scan[tid] + *num_ones; // value whose low weight bit is 1 therefore we shift a good number of 1 and we place from this index
		output[idx] = input[tid];               // put the value in the good position
	}
}

int main(int argc, char const* argv[]) {    
    int* test_input_host;			// input host
	int* test_input_device;			// input device

	int* test_output_host;			// output host
	int* test_output_device;	    // output device

	int* test_bits_host;			// bits host
	int* test_bits_device;		    // bits device

	int* test_reverse_bits_host;	// bits reverse from test_bits_host
	int* test_reverse_bits_device;  // bits reverse from test_bits_device

	int* nb_ones_host;				// number of ones host
	int* nb_ones_device; 			// number of ones device

	int* result_scan_host;          // result scan host
	int* result_scan_device;		// result scan device

	int* result_second_scan_host;   // result second scan host
	int* result_second_scan_device; // result second scan device

	cudaMallocHost((void**)&test_input_host,  8*sizeof(int));
	cudaMallocHost((void**)&test_output_host, 8*sizeof(int));
	cudaMallocHost((void**)&test_bits_host,   8*sizeof(int));
	cudaMallocHost((void**)&nb_ones_host,       sizeof(int));
	cudaMallocHost((void**)&result_scan_host, 		8*sizeof(int));
	cudaMallocHost((void**)&test_reverse_bits_host, 8*sizeof(int));
	cudaMallocHost((void**)&result_second_scan_host, 8*sizeof(int));

	cudaMalloc(&test_input_device,  8*sizeof(int));
	cudaMalloc(&test_output_device, 8*sizeof(int));
	cudaMalloc(&test_bits_device,   8*sizeof(int));
	cudaMalloc(&nb_ones_device,       sizeof(int));
	cudaMalloc(&result_scan_device, 	   8*sizeof(int));
	cudaMalloc(&test_reverse_bits_device,  8*sizeof(int));
	cudaMalloc(&result_second_scan_device, 8*sizeof(int));
	
	// ------------------- INITIALISATION ------------------ //
	
	test_input_host[0] =0; // 000
	test_input_host[1] =5; // 101
	test_input_host[2] =2; // 010
	test_input_host[3] =7; // 111
	test_input_host[4] =1; // 001
	test_input_host[5] =3; // 011
	test_input_host[6] =6; // 110
	test_input_host[7] =4; // 100

	cudaMemcpy(test_input_device, test_input_host, sizeof(int)*8, cudaMemcpyHostToDevice);

	int mask = 1;
	for(int x = 0; x < 3; x++){
		// ***************  DEBUG ************** //
		std::cout << "Orignal Array : ";
		for(int i = 0; i < 8; i++) {
			std::cout << test_input_host[i] << " ";
		}
		std::cout << std::endl ;
		
		cudaMemset(nb_ones_device, 0, sizeof(int)); // set 0 on the gpu for the numbers of ones
	
		getBit<<<1, 8>>>(test_input_device, test_bits_device, test_reverse_bits_device, nb_ones_device, mask); // Kernel Invoke
	
		cudaMemcpy(test_bits_host, test_bits_device, 8*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(nb_ones_host, nb_ones_device, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(test_reverse_bits_host, test_reverse_bits_device, sizeof(int)*8, cudaMemcpyDeviceToHost);
	
		// ***************  DEBUG ************** //
		std::cout << "Bit of value  : " ;
		for(int i = 0; i < 8; i++) {
			std::cout << test_bits_host[i] << " ";
        }
        std::cout << "\nBits reverse  : " ;
		for(int i = 0; i < 8; i++) {
			std::cout << test_reverse_bits_host[i] << " ";
		}
		std::cout << "\nNum of ones   : " <<  *nb_ones_host << std::endl ;

		cudaMemcpy(result_scan_device, test_bits_host, 8*sizeof(int), cudaMemcpyHostToDevice);
		blelloch_scan<<<1, 8>>>(result_scan_device, 8); // invoke blelloch scan on the array of bit
		cudaMemcpy(result_scan_host, result_scan_device, 8*sizeof(int), cudaMemcpyDeviceToHost);
		
		// ***************  DEBUG ************** //
		std::cout << "First scan    : " ;
		for(int i = 0; i < 8; i++) {
			std::cout << result_scan_host[i] << " ";
		}
        std::cout << std::endl;
        
		cudaMemcpy(result_second_scan_device, test_reverse_bits_host, 8*sizeof(int), cudaMemcpyHostToDevice);
		blelloch_scan<<<1, 8>>>(result_second_scan_device, 8); // invoke blelloch scan on the array of reversed bit
		cudaMemcpy(result_second_scan_host, result_second_scan_device, 8*sizeof(int), cudaMemcpyDeviceToHost);

		// ***************  DEBUG ************** //
		std::cout << "Second scan   : " ;
		for(int i = 0; i < 8; i++) {
			std::cout << result_second_scan_host[i] << " ";
		}
		std::cout << std::endl ;

		sort_array<<<1, 8>>>(test_input_device, test_output_device, test_bits_device, result_scan_device, result_second_scan_device, nb_ones_device);
		cudaMemcpy(test_output_host, test_output_device, sizeof(int)*8, cudaMemcpyDeviceToHost);
		
		// ***************  DEBUG ************** //
		std::cout << "\nStep [" << x + 1 << "/3] \nSorted GPU    : ";
		for(int i = 0; i < 8; i++) {
			std::cout << test_output_host[i] << " ";
		}
		std::cout << std::endl << std::endl;

		mask <<= 1; // change bit !

		cudaMemcpy(test_input_device, test_output_host,sizeof(int)*8,   cudaMemcpyHostToDevice);
	} 

	cudaFree(result_scan_device);
	cudaFree(test_reverse_bits_device);
	cudaFree(test_reverse_bits_host);
	cudaFree(test_bits_device);
	cudaFree(test_bits_host);
	cudaFree(result_scan_host);
	cudaFree(result_scan_device);
	cudaFree(nb_ones_host);
	cudaFree(nb_ones_device);
	cudaFree(test_output_device);
	cudaFree(test_input_device);
	cudaFree(test_output_host);
	cudaFree(test_input_host);

}