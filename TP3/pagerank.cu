#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include "cuStopwatch.cu"

#define COUNT (1<<23)
#define LINK_PER_PAGE 4
#define ERMIX 0.25f
#define MAXINT (4294967295.0f)
#define DAMPING 0.9f
#define EPSILON 0.00000001f
#define MAXPRCOUNT 16
#define INITPROJ 1024

#define ITERATION (COUNT*LINK_PER_PAGE)

/* ------------ Pagerank computation, GPU part ------------ */

  // TODO: fill in initial value for pagerank
  // Each thread, Initialize oldp array (length COUNT) with 1/COUNT for each element.

__global__ void pr_init_gpu(float* pr){
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < COUNT) {
        pr[tid] = 1 / static_cast<float>(COUNT);
    }
}

// Each thread, Initialize newp array (length COUNT) with (1-DAMPING)/COUNT for each element.
__global__ void pr_damping_gpu(float* pr){ 
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < COUNT) {
        pr[tid] = (1 - DAMPING)/ static_cast<float>(COUNT);
    }
}

//  Add the contribution of hyperlinks to each element of the newp array
__global__ void pr_iter_gpu(const uint2* links, const float* oldp, float* newp){
    // TODO: add contributions for each link for pagerank
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < ITERATION) {                                           // LINK_PER_PAGE*COUNT
        float val = DAMPING * oldp[links[tid].x] / LINK_PER_PAGE;   // new val computed
        atomicAdd(&newp[links[tid].y], val);                        // data race on the array, so we need to use atomic add() (on assembly instruction)
    }
}

// Check the convergence of each pairs : differences  of  all  pairs  of oldp[i] and newp[i] are  smaller  than EPSILON
__global__ void pr_conv_check_gpu(const float* oldp, const float* newp, uint32_t* conv){
    int t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    *conv = 0;                                                       // if the diff doesn't converge we change the convergance (*conv) to 1
    if(t_idx < COUNT && fabs(oldp[t_idx] - newp[t_idx]) > EPSILON) { // fabs = float absolute()
        *conv = 1;
    }

    // int hasNotConverged = fabs(oldp[t_idx] - newp[t_idx]) > EPSILON ? 1 : 0;
    // if(__any_sync(0xffffffff, hasNotConverged)) { // if one pair doesn't among each pairs converge return 
    //     return;
    // }
    // *conv = 0;

    // int hasConverged = fabs(oldp[t_idx] - newp[t_idx]) <= EPSILON ? 1 : 0; // if all pairs converge set conv to 0
    // if(__all_sync(0xffffffff, hasConverged)) {
    //     *conv = 0;
    // }
}

// control GPU computation, returns computation time (in seconds, not counting memory transfer time)
float pr_compute_gpu(const uint2* links, float* pr){
    cuStopwatch time;
    float* oldp;
    float* newp;
    uint32_t* conv_device;

    // Allocate arrays and conv_device variable in the device memory
    cudaMalloc((void**)&oldp, sizeof(float)*COUNT);
    cudaMalloc((void**)&newp, sizeof(float)*COUNT);
    cudaMalloc((void**)&conv_device, sizeof(uint32_t));


    int nb_thread = (1<<10);                     // number of max thread per grid (1024) 
    int nb_block = ceil(COUNT/nb_thread);        // calculate the right number of blocks to have one thread per page

    time.start();                                // start the timer

    pr_init_gpu<<< nb_block, nb_thread>>>(oldp); // initialize oldp array 

    uint32_t conv = 1;
    cudaMemcpyAsync(conv_device, &conv, sizeof(uint32_t), cudaMemcpyHostToDevice); 

    while(true){
        pr_iter_gpu<<<nb_block*LINK_PER_PAGE, nb_thread>>>(links, oldp, newp);          // contribution of hyperlinks to each element of the array newp   
        pr_conv_check_gpu<<<nb_block, nb_thread>>>(oldp, newp, conv_device);            // check all pairs for compute the convergence :  newp[i] oldp[i]
        cudaMemcpyAsync(&conv, conv_device, sizeof(uint32_t), cudaMemcpyDeviceToHost);  // put conv_device value in conv

        if(conv == 0){
            break;
        }
        cudaMemcpy(oldp, newp, sizeof(float)*COUNT, cudaMemcpyDeviceToDevice);  // put newp value in oldp
        pr_damping_gpu<<<nb_block, nb_thread>>>(newp);                          // initialize newp array : Step 2 of the algorithm
    }
    float result = time.stop()/1000;                                            // finish the timer
    cudaMemcpy(pr, newp, sizeof(float)*COUNT, cudaMemcpyDeviceToHost);          // pr point to newp (host array with computed pageranks)

    // Free the device memory in this scope
    cudaFree(oldp);
    cudaFree(newp);
    cudaFree(conv_device);

    return result;
}

/* ------------ Pagerank computation, CPU part ------------ */

// Like the GPU version, each thread, Initialize oldp array (length COUNT) with 1/COUNT for each element.
void pr_init_cpu(float* pr){
    // TODO: equivalence of pr_init_gpu on host
    for (int i = 0; i < COUNT; i++){
        pr[i] = 1 / static_cast<float>(COUNT);
    }
}

// Like the GPU version, each thread, Initialize newp array (length COUNT) with (1-DAMPING)/COUNT for each element.
void pr_damping_cpu(float* pr){
    // TODO: equivalence of pr_damping_gpu on host 
    for (int i = 0; i < COUNT; i++){
        pr[i] = (1 - DAMPING)/ static_cast<float>(COUNT);
    }
}

// Like the GPU version, add the contribution of hyperlinks to each element of the newp array
void pr_iter_cpu(const uint2* links, const float* oldp, float* newp){
    // TODO: equivalenc of pr_iter_gpu on host
    for (int i = 0; i < ITERATION; i++) {                               // COUNT * LINK_PER_PAGE
        newp[links[i].y] += DAMPING * (oldp[links[i].x] / LINK_PER_PAGE);
    }
}

// Like the GPU version, we check the convergence of each pairs : differences  of  all  pairs  of oldp[i] and newp[i] are  smaller  than EPSILON
void pr_conv_check_cpu(const float* oldp, const float* newp, uint32_t* conv){
    // TODO: equivalence of pr_conv_check_gpu on host
    for (int i = 0; i < COUNT; i++) {
        float val1 = oldp[i];
        float val2 = newp[i];
        if(abs((val1 - val2)) > EPSILON){ // Check  if  the  differences  of  all  pairs  of oldp [i] and newp[i] are smaller than EPSILON
            return;                 
        }
    }               
    *conv = 0;                      // stop condition, all pairs converged // all the value are less than EPSILON
}

float pr_compute_cpu(const uint2* links, float* pr){
    // TODO: equivalence of pr_compute_gpu on host
    float* oldp;
    cudaHostAlloc((void**)&oldp, sizeof(float)*COUNT, cudaHostAllocDefault);

    clock_t time = clock();         // start clock

    pr_init_cpu(oldp);              // initialize oldp array 
    pr_damping_cpu(pr);             // initialize pr array

    uint32_t conv = 1;
    while(true){  
        pr_iter_cpu(links, oldp, pr);           // calculate the contribution for each webpage
        pr_conv_check_cpu(oldp, pr, &conv);     // check if the pairs oldpr[i] and newpr[i] converge
        if(conv == 0){                          // quit the loop, the convergence is reached.
            break;
        }
        memcpy(oldp, pr, sizeof(float)*COUNT); // copy the oldp in pr array
        pr_damping_cpu(pr);                    // initialise pr array
    }

    cudaFreeHost(oldp);
    return (clock() - time)/CLOCKS_PER_SEC ; // stop clock
}

/* ------------ Random graph generation ------------ */

uint32_t randstate;

uint32_t myrand(){
    randstate ^= randstate << 13;
    randstate ^= randstate >> 17;
    randstate ^= randstate << 5;
    return randstate;
}

void seed(){
    randstate = time(NULL);
    for(int i = 0; i < 16; i++) myrand();
    return;
}

void randgen(uint2* links){
    uint32_t state = time(NULL);
    uint32_t *weight = (uint32_t*)malloc(sizeof(uint32_t) * COUNT);
    memset((void*)weight, 0, sizeof(uint32_t) * COUNT);
    uint32_t totalweight = 0;
    uint32_t lcnt = 0;
    
    // Initial five
    for(int i = 0; i < INITPROJ; i++){
        weight[i] = 1;
        for(int j = 0; j < 4; j++){
            links[lcnt].x = i;
            links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
            lcnt++;
        }
    }
    totalweight = INITPROJ;
    
    // Barabasi-Albert with Erdos-Renyi mix-in
    for(uint32_t i = INITPROJ; i < COUNT; i++){
        for(int k = 0; k < LINK_PER_PAGE; k++){
             if(myrand()/MAXINT < ERMIX){
                links[lcnt].x = i;
                links[lcnt].y = (uint32_t)(myrand()*(COUNT/MAXINT));
                lcnt++;
            }else{
                uint32_t randweight = (uint32_t)(myrand()/MAXINT*totalweight);
                uint32_t idx = 0;
                while(randweight > weight[idx]){
                    randweight -= weight[idx];
                    idx++;
                }
                links[lcnt].x = i;
                links[lcnt].y = idx;
                lcnt++;
                weight[idx]++;
                totalweight++;
            }
        }
    }
    return;
}

/* ------------ Main control ------------ */

void pr_extract_max(const float* pr, float* prmax, uint32_t* prmaxidx){
    for(int i = 0; i < MAXPRCOUNT; i++) prmax[i] = -1.0f;
    for(uint32_t i = 0; i < COUNT; i++){
        if(pr[i] > prmax[MAXPRCOUNT-1]){
            int ptr = 0;
            while(pr[i] <= prmax[ptr]) ptr++;
            float oldval, newval;
            uint32_t oldidx, newidx;
            newval = pr[i];
            newidx = i;
            for(int j = ptr; j < MAXPRCOUNT; j++){
                oldval = prmax[j];
                oldidx = prmaxidx[j];
                prmax[j] = newval;
                prmaxidx[j] = newidx;
                newval = oldval;
                newidx = oldidx;
            }
        }
    }
    return;
}

int main(){
    // Generating random network
    uint2* randlinks;
    cudaHostAlloc((void**)&randlinks, sizeof(uint2)*COUNT*LINK_PER_PAGE, cudaHostAllocDefault);
    seed();
    randgen(randlinks);
    printf("Finished generating graph\n\n");
    
    // Declaration of needed variables and arrays
    float prmax[MAXPRCOUNT];
    uint32_t prmaxidx[MAXPRCOUNT];
    float elapsed;
    float *pagerank;
    float check;
    cudaHostAlloc((void**)&pagerank, sizeof(float)*COUNT, cudaHostAllocDefault);
    
    // Processing by GPU
    elapsed = pr_compute_gpu(randlinks, pagerank);
    printf("GPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    printf("\n");
    
    // Processing by CPU
    elapsed = pr_compute_cpu(randlinks, pagerank);
    printf("CPU version, runtime %7.4fs\n", elapsed);
    check = 0.0f;
    for(uint32_t i = 0; i <COUNT; i++) check+=pagerank[i];
    printf("Deviation: %.6f\n", check);
    pr_extract_max(pagerank, prmax, prmaxidx);
    for(int i = 0; i < MAXPRCOUNT; i++){
        printf("Rank %d, index %u, normalized pagerank %8.7f\n", i, prmaxidx[i], prmax[i] / check);
    }
    
    // Free memory
    cudaFreeHost(randlinks);
    cudaFreeHost(pagerank);
	return 0;
}


/*void pr_iter_cpu(const uint2* links, const float* oldp, float* newp){
    // TODO: equivalenc of pr_iter_gpu on host
    for (int i = 0; i < COUNT; i++) {
        // int from = links[i].x;
        float compute_current_element = 0.0;

        for(int j = links[i].x ; j < links[i].y ; j++) { 
            compute_current_element += oldp[j] / LINK_PER_PAGE; // there is 4
        }

        newp[links[i].y] = (1 - DAMPING)/ static_cast<float>(COUNT) + DAMPING * compute_current_element;
    }
}*/

/*void pr_iter_cpu(const uint2* links, const float* oldp, float* newp){
    // TODO: equivalenc of pr_iter_gpu on host
    for (int i = 0; i < COUNT; i++) {
        // int from = links[i].x;
        float compute_current_element = 0.0;

        for(int j = 0; j < ITERATION ; j++) {                                // COUNT * LINK_PER_PAGE
            int to   = links[j].y;
            if(to == i){
                compute_current_element += oldp[links[j].x] / LINK_PER_PAGE; // there is 4
            }
        }
        newp[i] = DAMPING * compute_current_element;
    }
}*/