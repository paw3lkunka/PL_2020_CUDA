#ifndef _KIRSCH_HELPERS_INL
#define _KIRSCH_HELPERS_INL

#include <sys/types.h>

#define cudaMin(a, b) (((a) < (b)) ? (a) : (b)) 
#define cudaMax(a, b) (((a) > (b)) ? (a) : (b)) 

namespace helpers
{
    struct ImageData
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        u_char* data;
    };
    
    enum ConvolutionType
    {
        Default = 0,
        First = 1,
        Second = 2
    };

    /// @brief Kirsch convolution matrices
    constexpr int filter[8][3][3] =
    {
        {
            { 5, 5, 5 },
            { -3, 0, -3 },
            { -3, -3, -3 }
        },
        {
            { 5, 5, -3 },
            { 5, 0, -3 },
            { -3, -3, -3 }
        },
        {
            { 5, -3, -3 },
            { 5, 0, -3 },
            { 5, -3, -3 }
        },
        {
            { -3, -3, -3 },
            { 5, 0, -3 },
            { 5, 5, -3 }
        },
        {
            { -3, -3, -3 },
            { -3, 0, -3 },
            { 5, 5, 5 }
        },
        {
            { -3, -3, -3 },
            { -3, 0, 5 },
            { -3, 5, 5 }
        },
        {
            { -3, -3, 5 },
            { -3, 0, 5 },
            { -3, -3, 5 }
        },
        {
            { -3, 5, 5 },
            { -3, 0, 5 },
            { -3, -3, -3 }
        }
    };

    
} // namespace helpers


#endif // _KIRSCH_HELPERS_INL
