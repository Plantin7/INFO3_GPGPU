#include "SDL.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <complex>
#include <thrust/complex.h>

#define LEN 1024
#define LENSHIFT 10
#define ITERMAX 1024
#define getindex(i, j) (((i)<<LENSHIFT)+(j))
#define NCOLOR 64
#define NCOLORMASK 63

SDL_Window   *screen;
SDL_Renderer *ren;
SDL_Texture  *tex;
SDL_Surface  *mysurf;

uint32_t iterscpu[LEN*LEN];
uint32_t colors[NCOLOR+1];
uint32_t* iters;


/* 
    ---- Kernel definition ----
    This method does exactly the same as the CPU method
    This method is used to determine whether a value is in or out of the Mandelbrot set. 
    We run through the formula Zn + 1 = Zn ^ 2 + C up to itermax (this is the depth of the previous image) 
    and at each iteration, we check if the real and imaginary part of Z is greater than 2; 
    if this happens we return the last iteration (which will give the color of the corresponding pixel).
    So we'll return the iteration in which it went out of bounds, and then interpret this number as a color. 
    If it completes the loop without going out of bounds, we will give it the color black
*/   
__device__ uint32_t compute_iteration_gpu(double ci, double cj, uint32_t itermax) 
{
    // We cannot use the complexe number of std, so we programming it manually
    double zRe = 0;     // real number     
    double zIm = 0;     // imaginary number
    
    double zReRes = 0;  
    double zImRes = 0;

    uint32_t iter = 0; // iteration

    for (iter = 0; iter < itermax; iter++){ 
        // Compute the new imaginary and real part of Z 
        zIm =  zRe * zIm;
        zIm += zIm + cj;

        zRe = zReRes - zImRes + ci;
        zReRes = zRe * zRe;
        zImRes = zIm * zIm;
        
        if (zReRes + zImRes >= 4.0) break; // greater than 2, so the value is out of the Mandelbrot set
    }
    
    return iter;
}

/* 
  OLD METHOD :
  The compute_iteration_gpu method is similar to the compute_iteration_cpu method.
  using thrust api !
*/

/*__device__ uint32_t compute_iteration_gpu(double ci, double cj, uint32_t itermax) 
{
    thrust::complex<double> z(0);
    thrust::complex<double> c(ci, cj);

    uint32_t iter = 0;
    for (iter = 0; iter < itermax; iter++) {
        z = (z * z) + c;
        if(abs(z) >= 2) break;
    }
    return iter;
}
*/

/* 
    CPU Version
    This method is used to determine whether a value is in or out of the Mandelbrot set. 
    We run through the formula Zn + 1 = Zn ^ 2 + C up to itermax (this is the depth of the previous image) 
    and at each iteration, we check if the real and imaginary part of Z is greater than 2; 
    if this happens we return the last iteration (which will give the color of the corresponding pixel).
    So we'll return the iteration in which it went out of bounds, and then interpret this number as a color. 
    If it completes the loop without going out of bounds, we will give it the color black
*/ 

int compute_iteration_cpu(double ci, double cj, uint32_t itermax)
{
    // use complex type for complexe number from std
    std::complex<double> z(0);
    std::complex<double> c(ci, cj);

    uint32_t iter = 0;
    for (iter = 0; iter < itermax; iter++) {
        // Compute the new imaginary and real part of Z 
        z = (z * z) + c;
        if(abs(z) >= 2) break; // greater than 2, so the value is out of the Mandelbrot set
    }
    return iter;
}

/*
    Mandelbrot using CPU
    This function compute all iteration of complexe number, for each pixel i and j and stock the result into the arr.
    X and Y parameter corresponds to the starting point of the iteration, i.e. at the top left of the screen (coordinate (0, 0))
    The parameter delta is used to know the next pixel to compute the itération.
    The variable itermax (1024) is there to give the maximum depth to calculate for the image
*/
void iterate_cpu(uint32_t *arr, double x, double y, double delta, uint32_t itermax)
{
    for (int i = 0; i < LEN * LEN; i++) {
        int xi = i % LEN;                   // index i, 0 to 1023, avoid to used two loop
        int yj = i / LEN;                   // index j, 0 to 1023, avoid to used two loop
        double ci = x + (yj * delta);
        double cj = y - (xi * delta);
        arr[getindex(xi, yj)] = compute_iteration_cpu(ci, cj, itermax); // compute itération 
    }
    return;
}

/*
    ---- Kernel definition ----
    Mandelbrot using GPU
    Unlike the CPU method, we do not need to do a loop, 
    because it is handled by the threads of each grid, one thread will take care of the calculation for each iteration.
    X and Y parameter corresponds to the starting point of the iteration, i.e. at the top left of the screen (coordinate (0, 0))
    The parameter delta is used to know the next pixel to compute the itération.
    The variable itermax (1024) is there to give the maximum depth to calculate for the image
*/

__global__ void iterate_gpu(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    int tId = blockDim.x * blockIdx.x + threadIdx.x ;                      

    int xi = tId % LEN;             // index i, 0 to 1023
    int yj = tId / LEN;             // index j, 0 to 1023
    double ci = x + (yj * delta);
    double cj = y - (xi * delta);
    arr[getindex(xi, yj)] = compute_iteration_gpu(ci, cj, itermax); // compute the iteration
    
    return;
}

/*
    This function call the kernel method (using __global___)
*/
void kernel_call(uint32_t* arr, double x, double y, double delta, uint32_t itermax){
    uint32_t thread_max = 1024;                                                  // number max of thread per grid
    int number_of_blocks = ceil((LEN*LEN)/thread_max);                           // calculate ideal number grid based on screen size
    iterate_gpu<<<number_of_blocks, thread_max, 0>>>(arr, x, y, delta, itermax); // Kernel invocation with 1024 threads
    cudaDeviceSynchronize();
    return;
}

void generate_colors(const SDL_PixelFormat* format){
    double h = 0.0;
    for(int i=0; i<NCOLOR; i++){
        int ph = h / 60;
        float f = (h/60.0 - ph);
        int v = 255;
        int p = 64;
        int q = (int)(255*(1 - f*0.75f));
        int t = (int)(255*(0.25f + f*0.75f));
        switch(ph){
            case 0:
            colors[i] = SDL_MapRGB(format, v, t, p);
            break;
            case 1:
            colors[i] = SDL_MapRGB(format, q, v, p);
            break;
            case 2:
            colors[i] = SDL_MapRGB(format, p, v, t);
            break;
            case 3:
            colors[i] = SDL_MapRGB(format, p, q, v);
            break;
            case 4:
            colors[i] = SDL_MapRGB(format, t, p, v);
            break;
            case 5:
            colors[i] = SDL_MapRGB(format, v, p, q);
            break;
            default:
            break;
        }
        h += 360.0/NCOLOR;
    }
    colors[NCOLOR] = SDL_MapRGB(format, 0, 0, 0);
    return;
}

int main(int argc, char** argv){
    SDL_Event e;
    bool usegpu = false;
    if(argc > 1){
        usegpu = (strcmp(argv[1], "gpu") == 0);
    }
    uint32_t* gpuarray;
    uint32_t* hostarray;

    // Initialize SDL
    if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
    atexit(SDL_Quit);
    // Create window
    screen = SDL_CreateWindow("Mandelbrot", 
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        LEN, LEN, SDL_WINDOW_SHOWN);
    if ( screen == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }

    // Initialize CUDA
    if(usegpu){
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaMalloc((void**)&gpuarray, LEN*LEN*sizeof(uint32_t));
        cudaHostAlloc((void**)&hostarray, LEN*LEN*sizeof(uint32_t), cudaHostAllocDefault);
    }

    // Create renderer and texture
    SDL_PixelFormat* fmt = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA32);
    generate_colors(fmt);
    ren = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex = SDL_CreateTexture(ren, fmt->format, SDL_TEXTUREACCESS_STREAMING, LEN, LEN);

    // Timing
    float totaltime = 0.0f;
    uint32_t frames = 0;

    // Window for Mandelbrot
    double targetx = -0.743643887037158704752191506114774;
    double targety = 0.131825904205311970493132056385139;
    double centerx = 0.0;
    double centery = 0.0;
    double delta = 4.0/LEN;
    const double scale = 0.94;
    uint32_t itermax = 32;
    const uint32_t iterstep = 8;

    while(true){
        bool flag = false;
        while(SDL_PollEvent(&e)){
            if(e.type==SDL_QUIT){
                flag = true;
            }
        }
        if(flag) break;
        clock_t t;
        float tsec;
        t = clock();
        // renderer
        if(!usegpu){
            iterate_cpu(iterscpu, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            iters = iterscpu;
        }else{
            kernel_call(gpuarray, centerx - delta*LEN/2, centery + delta*LEN/2, delta, itermax);
            cudaMemcpyAsync(hostarray, gpuarray, LEN * LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            iters = hostarray;
        }

        int len = LEN;
        uint32_t* surf = NULL;
        SDL_LockTexture(tex, NULL, (void**)(&surf), &len);
        for(uint32_t i=0; i<LEN*LEN; i++){
            if (iters[i] < itermax){
                surf[i] = colors[iters[i]&NCOLORMASK];
            }else{
                surf[i] = colors[NCOLOR];
            }
        }
        SDL_UnlockTexture(tex);
        SDL_RenderClear(ren);
        SDL_RenderCopy(ren, tex, NULL, NULL);
        SDL_RenderPresent(ren);
        centerx = targetx + (centerx - targetx)*scale;
        centery = targety + (centery - targety)*scale;
        delta *= scale;
        itermax += iterstep;
        t = clock() - t;
        tsec = ((float)t)/CLOCKS_PER_SEC;
        totaltime += tsec;
        tsec = 1.0f/60 - tsec;
        if(tsec > 0) SDL_Delay((uint32_t)(tsec*1000));
        frames++;
        if(frames>=530) break;
    }

    char s[100];
    sprintf(s, "Average FPS: %.1f\nFrame count: %u", frames/totaltime, frames);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Benchmark", s, screen);
    SDL_FreeFormat(fmt);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(screen);
    if(usegpu){
        cudaFree(gpuarray);
        cudaFreeHost(hostarray);
    }
    exit(0);
}