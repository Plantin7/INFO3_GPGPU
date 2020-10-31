nvcc -o mandelbrot.exe -O2 --include-path /usr/include/SDL2 --compiler-options -D_REENTRANT  --linker-options -lSDL2 mandelbrot.cu
