#include "SDL_image.h"
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cuStopwatch.cu"

#include <iterator>
#include <iostream>

enum conv_t{
    CONV_IDENTITY,
    CONV_EDGE,
    CONV_SHARP,
    CONV_GAUSS
};

SDL_Window *screen;
SDL_Window *screen_res;
SDL_Renderer *ren;
SDL_Renderer *ren_res;
SDL_Texture *tex;
SDL_Texture *tex_res;
SDL_Surface *surf;

int32_t width, height;
float filter[9];

float __constant__ filter_device[9];

// Check if the index on the row is reachable
__device__ int checkRowPixels(int row, int currentIdx, int width, int height) {
    if(row < 0 || row == height) {
        return 0;
    } 
    return width * 3 * currentIdx;
}

// Check if the index on the column is reachable
__device__ int checkColumnPixels(int column, int currentIdy, int width) {
    if(column < 0 || column == width) {
        return 0;
    } 
    return currentIdy * 3;
}

// Compute and apply on the picture the given filter 
__device__ char compute_filter(const unsigned char* src, int tidx, int tidy, int w, int h, int32_t color) {
    int result = 0;
    int row    = tidx / w; // representation of 2D coordinate x
    int column = tidy / h; // representation of 2D coordinate y

    for(int i = -1; i <= 1 ; i++){
        row = checkRowPixels(row + i, i, w, h);             // check if row case we want to retrieve is outside the table
        for(int j = -1; j <= 1; j++) {
            column = checkColumnPixels(column + j, j, w);   // check if column case we want to retrieve is outside the table
            result += filter_device[(i+1)*3+(j+1)] * src[tidx*3 + tidy*3 + row + column + color];  // tids * 3 / three colors represented 
                                                                                                   // operation to get the good index (0, 1, 2...)
        }
    }

    return (result < 0 ?  0 : (result > 255 ? 255 : result)); // the value of a color of a pixel can range from 0 to 255
                                                              // so if we want a nice image we have to truncate the value
}
/**
    Apply the given filter on the given image stored in the global memory
*/
__global__ void conv_global(const unsigned char* src, unsigned char* dest, int32_t w, int32_t h){
    int tIdx = blockDim.x * blockIdx.x + threadIdx.x;       // identifiant of thread x 
    int tIdy = blockDim.y * blockIdx.y + threadIdx.y;       // identifiant of thread y

    if(tIdx < w*h*3) {                                      // check if we are in the image to compute the filter
        for(int color = 0; color < 3; color++) {            // loop on the three colors
            dest[tIdx*3 + color] = compute_filter(src, tIdx, tIdy, w, h, color); // compute and store in the dest the value of each pixel after passed the filter
        }
    }
}

// cudaAlloc/ cudaMemoryCopy from the host and back to the host

float conv_global_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // todo: write the code that manages memory (global memory) and invokes the kernel conv_global, it should return the running time
    int size = w * h * 3;
    unsigned char *pixels_dev, *output_dev, *host_pixels;
    cuStopwatch timer;
    // start the timer
    timer.start();

    cudaHostAlloc((void **)&host_pixels, size * sizeof(unsigned char), cudaHostAllocDefault); // Allocate Host memory of the image from the device / get the image from the host and transfer to the gpu
    cudaMalloc((void **)&pixels_dev, size * sizeof(unsigned char));                           // allocate the original image for the GPU
    cudaMalloc((void **)&output_dev, size * sizeof(unsigned char));                           // allocate image filtered for the GPU

    memcpy(host_pixels, pixels, size * sizeof(unsigned char));                                      // copy the content picture on the host
    cudaMemcpy(pixels_dev, host_pixels, size * sizeof(unsigned char), cudaMemcpyHostToDevice);      // copy the content of the host on the allocated array
                                                                                                    // allow to transfer the picture from the host memory to the device memory


    int block_number = ceil((h * w) / 1024);                                                        // ideal number of thread per block 
    conv_global<<<block_number, 1024>>>(pixels_dev, output_dev, w, h);                              // applying the filter

    cudaMemcpy(host_pixels, output_dev, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);      // transfer the new picture filtered from the device to the host memory
    memcpy(pixels, host_pixels, size);

    // Free Memory allocated
    cudaFreeHost(host_pixels);
    cudaFree(pixels_dev);
    cudaFree(output_dev);

    return timer.stop(); // stop the timer
}


__global__ void conv_texture(cudaTextureObject_t src, unsigned char* dest, int32_t w, int32_t h){
    // todo: write a kernel to apply the given filter on the given image stored as a texture
}

float conv_texture_gpu(unsigned char* pixels, int32_t w, int32_t h){
    // todo: write the code that manages memory (texture memory) and invokes the kernel conv_global, it should return the running time
    return 0.0;
}

int main(int argc, char** argv){
    SDL_Event event;
    bool withtex = false;
    
    // Initialize SDL
    if( SDL_Init(SDL_INIT_VIDEO) < 0 ) {
        fprintf(stderr, "Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }
	atexit(SDL_Quit);
    
    if(argc == 1){
        exit(1);
    }
    
    // Read image and option
    IMG_Init(IMG_INIT_PNG);
    surf = IMG_Load(argv[1]);
    if(surf == NULL){
        fprintf(stderr, "Error loading image.\n");
        exit(1);
    }
    width = surf->w;
    height = surf->h;
    SDL_SetSurfaceRLE(surf, 1);
    
    // Initialize involution kernel
    conv_t conv_type;
    if(argc >= 3){
        if (strcmp(argv[2], "identity") == 0) conv_type = CONV_IDENTITY;
        else if (strcmp(argv[2], "edge") == 0) conv_type= CONV_EDGE;
        else if (strcmp(argv[2], "sharp") == 0) conv_type= CONV_SHARP;
        else if (strcmp(argv[2], "gauss") == 0) conv_type = CONV_GAUSS;
        else conv_type = CONV_IDENTITY;
    }
    switch(conv_type){
        case CONV_EDGE:
            filter[0] = -1; filter[1] = -1; filter[2] = -1; 
            filter[3] = -1; filter[4] = 8; filter[5] = -1; 
            filter[6] = -1; filter[7] = -1; filter[8] = -1; 
            break;
        case CONV_SHARP:
            filter[0] = 0; filter[1] = -1; filter[2] = 0; 
            filter[3] = -1; filter[4] = 5; filter[5] = -1; 
            filter[6] = 0; filter[7] = -1; filter[8] = 0; 
            break;
        case CONV_GAUSS:
            filter[0] = 1.0f/16; filter[1] = 1.0f/8; filter[2] = 1.0f/16; 
            filter[3] = 1.0f/8; filter[4] = 1.0f/4; filter[5] = 1.0f/8; 
            filter[6] = 1.0f/16; filter[7] = 1.0f/8; filter[8] = 1.0f/8; 
            break;
        default:
            filter[0] = 0; filter[1] = 0; filter[2] = 0; 
            filter[3] = 0; filter[4] = 1; filter[5] = 0; 
            filter[6] = 0; filter[7] = 0; filter[8] = 0; 
            break;
    }
    cudaMemcpyToSymbolAsync(filter_device, filter, sizeof(float)*9, 0, cudaMemcpyHostToDevice);
    
    if(argc >= 4){
        if(strcmp(argv[3], "texture") == 0) withtex = true;
    }
    
    // Create window
	screen = SDL_CreateWindow("Original", 
                        100,
                        100,
                        width, height, SDL_WINDOW_SHOWN);
    if ( screen == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    screen_res = SDL_CreateWindow("Filtered", 
                        300,
                        300,
                        width, height, SDL_WINDOW_SHOWN);
    if ( screen_res == NULL ) {
        fprintf(stderr, "Couldn't set up window: %s\n", SDL_GetError());
        exit(1);
    }
    
    // Initialize CUDA
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    
    // Create renderer and texture
    ren = SDL_CreateRenderer(screen, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex = SDL_CreateTextureFromSurface(ren, surf);
    
    // Show image
    SDL_RenderCopy(ren, tex, NULL, NULL);
    SDL_RenderPresent(ren);
    
    // Compute
    SDL_LockSurface(surf);
    float elapsed;
    if(withtex){
        elapsed = conv_texture_gpu((unsigned char*)surf->pixels, width, height);
    }else{
        elapsed = conv_global_gpu((unsigned char*)surf->pixels, width, height);
    }
    SDL_UnlockSurface(surf);
    
    // Show computed image
    ren_res = SDL_CreateRenderer(screen_res, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    tex_res = SDL_CreateTextureFromSurface(ren_res, surf);
    SDL_RenderCopy(ren_res, tex_res, NULL, NULL);
    SDL_RenderPresent(ren_res);
    SDL_FreeSurface(surf);
    
    while (1) {
        SDL_WaitEvent(&event);
        if ((event.type == SDL_QUIT) || ((event.type == SDL_WINDOWEVENT) && (event.window.event == SDL_WINDOWEVENT_CLOSE))) break;
    }
    
    char s[100];
    sprintf(s, "Kernel execution time: %.4fms", elapsed);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Timing", s, screen);
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(screen);
    SDL_DestroyWindow(screen_res);
    exit(0);
}