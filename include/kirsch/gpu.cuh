#ifndef _KIRSCH_GPU_CUH
#define _KIRSCH_GPU_CUH

#include <sys/types.h>
#include <helper_cuda.h>

#include "kirsch/helpers.inl"

namespace gpu
{
    constexpr int MASK_WIDTH = 14;
    constexpr int TILE_WIDTH = 16;
    constexpr int O_TILE_WIDTH = 14;
    constexpr int BLOCK_WIDTH = O_TILE_WIDTH + MASK_WIDTH - 1;
    constexpr int SM_WIDTH = TILE_WIDTH + MASK_WIDTH - 1;
    constexpr int WIDTH_SQUARE = TILE_WIDTH * TILE_WIDTH;
    constexpr int NUM_LOADS = static_cast<int>( static_cast<float>(SM_WIDTH * SM_WIDTH) * static_cast<float>(WIDTH_SQUARE) ) + 1;
    constexpr int DIVIDE_WIDTH = TILE_WIDTH + 2;

    __global__ void convolution(u_char* I, const int* __restrict__ M, u_char* P, int channels, int width, int height);
    __global__ void convolution2(u_char* I, const int* __restrict__ M, u_char* P, int channels, int width, int height);

    void edgeDetection
    (
        void* input, 
        void* mask, 
        void* output, 
        u_int channels, 
        u_int width, 
        u_int height, 
        helpers::ConvolutionType convolutionType = helpers::ConvolutionType::First
    );
} // namespace gpu


#endif // _KIRSCH_GPU_CUH
