#include <iostream>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

//#include "kirsch.hpp"
#include "cpu.hpp"
#include "gpu.cuh"
#include "helpers.inl"

int main(int argc, char** argv)
{
    // std::ios_base::sync_with_stdio(false);
    // std::cin.tie(NULL);

    if(argc != 4)
    {
        std::cout << "Invalid argument count." << '\n';
        return -1;
    }

	// GPU

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = findCudaDevice( argc, const_cast<const char**>(argv) );

    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, dev) );
    std::cout << "CUDA device " << deviceProp.name << " has " << deviceProp.multiProcessorCount
        << " multi-processors, compute " << deviceProp.major << '.' << deviceProp.minor << '\n';

    StopWatchInterface* host_timer = nullptr;
	sdkCreateTimer(&host_timer);
	double timerSeconds = 0;

	helpers::ImageData image;
	image.data = stbi_load(argv[1], &image.width, &image.height, &image.channels, 0);

	u_int byteCount = image.width * image.height * image.channels * sizeof(u_char);

	u_char* host_gpu_output = new u_char[byteCount];
	std::copy(image.data, image.data + byteCount, host_gpu_output);

	u_char* device_gpu_input;
	int* device_gpu_mask;
	u_char* device_gpu_output;
	checkCudaErrors( cudaMalloc( reinterpret_cast<void**>(&device_gpu_input), byteCount ) );
	checkCudaErrors( cudaMalloc( reinterpret_cast<void**>(&device_gpu_mask), 8 * 9 * sizeof(int) ) );
	checkCudaErrors( cudaMalloc( reinterpret_cast<void**>(&device_gpu_output), byteCount ) );

	checkCudaErrors( cudaMemcpy(device_gpu_input, image.data, byteCount, cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaMemcpy(device_gpu_mask, helpers::filter, 8 * 9 * sizeof(int), cudaMemcpyHostToDevice) );

	checkCudaErrors( cudaDeviceSynchronize() );

	sdkResetTimer(&host_timer);
	sdkStartTimer(&host_timer);

	gpu::edgeDetection
	(
		device_gpu_input, 
		device_gpu_mask, 
		device_gpu_output, 
		image.channels, 
		image.width, 
		image.height
	);
	checkCudaErrors( cudaMemcpy(host_gpu_output, device_gpu_output, byteCount, cudaMemcpyDeviceToHost) );

	sdkStopTimer(&host_timer);
	timerSeconds = 1.0e-3 * static_cast<double>( sdkGetTimerValue(&host_timer) );

	std::cout << "Kirsch edge detection using GPU" << ' '
		<< "time: " << timerSeconds << " sec," << ' '
		<< "speed: " << ( static_cast<double>(byteCount) * 1.0e-6 / timerSeconds) << '\n';

	checkCudaErrors( cudaFree(device_gpu_input) );
	checkCudaErrors( cudaFree(device_gpu_mask) );
	checkCudaErrors( cudaFree(device_gpu_output) );

	cudaDeviceReset();

	// CPU

	u_char* host_cpu_output = new u_char[byteCount];
	std::copy(host_cpu_output, host_cpu_output + byteCount, image.data);

	sdkResetTimer(&host_timer);
	sdkStartTimer(&host_timer);

	cpu::edgeDetection
	(
		image.data, 
		reinterpret_cast<const int*>(helpers::filter), 
		host_cpu_output,
		image.channels,
		image.width,
		image.height
	);

	sdkStopTimer(&host_timer);
	timerSeconds = 1.0e-3 * static_cast<double>( sdkGetTimerValue(&host_timer) );

	std::cout << "Kirsch edge detection using CPU" << ' '
		<< "time: " << timerSeconds << " sec," << ' '
		<< "speed: " << ( static_cast<double>(byteCount) * 1.0e-6 / timerSeconds) << '\n';

	sdkDeleteTimer(&host_timer);

	// Writing output files

	stbi_write_jpg(argv[2], image.width, image.height, image.channels, host_cpu_output, 100);
	stbi_write_jpg(argv[3], image.width, image.height, image.channels, host_gpu_output, 100);

	delete [] host_cpu_output;
	delete [] host_gpu_output;

    return 0;
}
