#include "gpu.cuh"

__global__ void gpu::convolution(u_char* I, const int* __restrict__ M, u_char* P, int channels, int width, int height)
{
    __shared__ int Ns[SM_WIDTH][SM_WIDTH][3];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int size = width * height;

    for(int i = 0; i < NUM_LOADS; i++)
    {
        int A = ty * TILE_WIDTH + tx + (i * WIDTH_SQUARE);
        
        int sx = A % DIVIDE_WIDTH;
        int sy = A / DIVIDE_WIDTH;
        
        int iX = (bx * TILE_WIDTH) + sx - 1;
        int iY = (by * TILE_WIDTH) + sy - 1;

        if((sx < SM_WIDTH) && (sy < SM_WIDTH))
        {
            if((iY >= 0) && (iY < height) && (iX >= 0) && (iX < width))
            {
                Ns[sy][sx][0] = I[iY * width + iX];
				Ns[sy][sx][1] = I[iY * width + iX + size];
				Ns[sy][sx][2] = I[iY * width + iX + 2 * size];
            }
            else
            {
                Ns[sy][sx][0] = 0;
				Ns[sy][sx][1] = 0;
				Ns[sy][sx][2] = 0;
            }
        }
    }

    __syncthreads();

    int row_o = by * TILE_WIDTH + ty;
    int col_o = bx * TILE_WIDTH + tx;

    if(row_o < height && col_o < width)
	{
		int max_sum = 0;
		int max_sum1 = 0;
		int max_sum2 = 0;
		
        for (int m = 0; m < 8; ++m)
		{
			int sum = 0;
			int sum1 = 0;
			int sum2 = 0;

			for (int i = 0; i < MASK_WIDTH; ++i)
			{
				for (int j = 0; j < MASK_WIDTH; ++j)
				{
					sum += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][0];
					sum1 += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][1];
					sum2 += *(M + m * 9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx][2];
				}
			}

			max_sum = cudaMax(sum, max_sum);
			max_sum1 = cudaMax(sum1, max_sum1);
			max_sum2 = cudaMax(sum2, max_sum2);
		}

		P[(row_o * width + col_o)] = cudaMin( cudaMax(max_sum / 8, 0), 255 );
		P[(row_o * width + col_o) + size] = cudaMin( cudaMax(max_sum1 / 8, 0), 255 );
		P[(row_o * width + col_o) + 2 * size] = cudaMin( cudaMax(max_sum2 / 8, 0), 255 );

	}

	__syncthreads();
}

__global__ void gpu::convolution2(u_char* I, const int* __restrict__ M, u_char* P, int channels, int width, int height)
{
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
	
	int tx = threadIdx.x; 
    int ty = threadIdx.y;
	
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
	int col_o = blockIdx.x * O_TILE_WIDTH + tx;

	int row_i = row_o - 1;
	int col_i = col_o - 1;

	int size = width * height;

	for (int layer = 0; layer < channels; ++layer) 
    {
		int max_sum = 0;

		if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
		{
			Ns[ty][tx] = I[row_i * width + col_i + layer * size];
		}
		else
		{
			Ns[ty][tx] = 0.f;
		}

		__syncthreads();
		
        if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
		{
			for (int m = 0; m < 8; ++m)
			{
				int sum = 0;
				for (int i = 0; i < MASK_WIDTH; ++i)
				{
					for (int j = 0; j < MASK_WIDTH; ++j)
					{
						sum += *(M + m*9 + i*MASK_WIDTH + j) * Ns[i + ty][j + tx];
					}
				}

				max_sum = sum > max_sum ? sum : max_sum;
			}

			if (row_o < height && col_o < width)
			{
				P[(row_o * width + col_o) + layer * size] = cudaMin( cudaMax(max_sum / 8, 0), 255 );
			}
		}

		__syncthreads();
	}
}

void gpu::edgeDetection(void* input, void* mask, void* output, u_int channels, u_int width, u_int height, helpers::ConvolutionType convolutionType)
{
    switch (convolutionType)
    {
    case helpers::ConvolutionType::First:
        {
            dim3 dimGrid(((width - 1) / TILE_WIDTH) + 1, ((height - 1) / TILE_WIDTH) + 1, 1);
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

            convolution << <dimGrid, dimBlock>> >
            (
                static_cast<u_char*>(input),
                static_cast<int*>(mask),
                static_cast<u_char*>(output),
                static_cast<int>(channels),
                static_cast<int>(width),
                static_cast<int>(height)
            );
        }
        break;

    case helpers::ConvolutionType::Second:
        {
            dim3 dimGrid(((width - 1) / O_TILE_WIDTH) + 1, ((height - 1) / O_TILE_WIDTH) + 1, 1);
            dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

            convolution2 << <dimGrid, dimBlock >> >
            (
                static_cast<u_char*>(input),
                static_cast<int*>(mask),
                static_cast<u_char*>(output),
                static_cast<int>(channels),
                static_cast<int>(width),
                static_cast<int>(height)
            );
        }
        break;
    }

    getLastCudaError("Kirsch edge detection has failed" + '\n');
    cudaDeviceSynchronize();
}
