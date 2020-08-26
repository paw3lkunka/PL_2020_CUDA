#ifndef _KIRSCH_CPU_HPP
#define _KIRSCH_CPU_HPP

#include <sys/types.h>

namespace cpu
{
    constexpr int MASK_WIDTH = 3;

    void edgeDetection
    (
        const u_char* input,
        const int32_t* mask,
        u_char* output,
        const u_int channels,
        const u_int width,
        const u_int height,
        bool colorInterleave = false
    );
} // namespace cpu


#endif // _KIRSCH_CPU_HPP
