#include "cutil_math.h"
#define usint (unsigned int)

// From https://nullprogram.com/blog/2018/07/31/
__device__ __forceinline__ usint hash(usint a)
{
    a ^= a >> 16;
    a *= 0x7feb352d;
    a ^= a >> 15;
    a *= 0x846ca68b;
    a ^= a >> 16;
    return a;
}

__device__ __forceinline__ usint hash2(usint a, usint b) {
    return hash(a) ^ hash(b);
}

__device__ __forceinline__ usint hash3(usint a, usint b, usint c) {
    return hash(a) ^ hash(b) ^ hash(c);
}

__device__ __forceinline__ usint hash4(usint a, usint b, usint c usint d) {
    return hash2(a,b) ^ hash2(c,d);
}

__device__ __forceinline__ float hash2intfloat(usint a, usint b) {
    return __uint2float_rd( hash2(a,b) );
}

//const float = 2.3283064365387e-10;