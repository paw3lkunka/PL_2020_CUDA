#include "cpu.hpp"

#include <algorithm>

void cpu::edgeDetection(const u_char* input, const int32_t* mask, u_char* output, const u_int channels, const u_int width, const u_int height, bool colorInterleave)
{
    u_int size = height * width;
    u_int i = 0;
    u_int j = 0;
    u_int layer = 0;

    for(; layer < channels; layer++)
    {
        for(; i < height; i++)
        {
            for(; j < width; j++)
            {
                int max_sum = 0;

                for(u_int m = 0; m < 8; m++)
                {
                    int sum = 0;
                    int sum1;
                    int sum2;
                    int sum3;

                    if((i >= 1) && (i + 1 < height) && (j >= 1) && (j + 1 < width))
                    {
                        for(int k = -1; k < 2; k++)
                        {
                            for(int l = -1; l < 2; l++)
                            {
                                if(colorInterleave)
                                {
                                    sum += *(mask + (m * 9) + ((k + 1) * channels) * (l + 1)) * static_cast<int>( input[ (i + k) * width + (j + l) + (layer * size) ] );
                                }
                                else
                                {
                                    sum += *(mask + (m * 9) + ((k + 1) * MASK_WIDTH)  + (l + 1)) * static_cast<int>( input[ ((i + k) * width + (j + l)) + (layer * size) ] );
                                }
                            }
                        }
                    }
                    else
                    {
                        for(int k = -1; k < 2; k++)
                        {
                            for(int l = -1; l < 2; l++)
                            {
                                if((i + k >= 0) && (i + k < height) && (j + l >= 0) && (j + l < width))
                                {
                                    if(colorInterleave)
                                    {
                                        sum += *(mask + (m * 9) + ((k + 1) * channels) + (l + 1)) * static_cast<int>( input[ ((i + k) * width + (j + l)) + (layer * size) ] );
                                    }
                                    else
                                    {
                                        sum += *(mask + (m * 9) + ((k + 1) * MASK_WIDTH) + (l + 1)) * static_cast<int>( input[ ((i + k) * width + (j + l)) + (layer * size) ] );
                                    }
                                }
                            }
                        }
                    }

                    max_sum = std::max(max_sum, sum);
                }

                if(colorInterleave)
                {
                    output[ (i * width + j) * channels + layer ] = std::min( std::max(max_sum / 8, 0), 255 );
                }
                else
                {
                    output[ (i * width + j) + (layer * size) ] = std::min( std::max(max_sum / 8, 0), 255 );
                }
            }
        }
    }
}