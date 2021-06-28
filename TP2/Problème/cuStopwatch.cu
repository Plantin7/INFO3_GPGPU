#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class cuStopwatch{
    // todo: add your internal data structure, all in private
	private:
		cudaEvent_t startEvent;
		cudaEvent_t stopEvent;
		bool isStartEventStarted;

    public:
        cuStopwatch();
        ~cuStopwatch();
        void start();
        float stop();
};

cuStopwatch::cuStopwatch(){
    // todo: constructor
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    isStartEventStarted = false;
}

cuStopwatch::~cuStopwatch(){
    // todo: destructor
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void cuStopwatch::start(){
    // todo: start the stopwatch, and ignore double start
    if(!isStartEventStarted){
    	cudaEventRecord(startEvent);
    	isStartEventStarted = true;
    }
}

float cuStopwatch::stop(){
    // todo: stop the stopwatch and return elapsed time, ignore invalid stops (e.g. stop when not yet started or double stop)
    if(!isStartEventStarted){
    	return -1.0;
    }
    
    cudaEventSynchronize(startEvent);
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

	float ms;
    cudaEventElapsedTime(&ms, startEvent, stopEvent);
    return ms;
}