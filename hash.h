#include "cutil_math.h"

// From https://nullprogram.com/blog/2018/07/31/
__device__ __forceinline__ unsigned hash(unsigned a) 
{
    a ^= a >> 16;
    a *= 0x7feb352d;
    a ^= a >> 15;
    a *= 0x846ca68b;
    a ^= a >> 16;
    return a;
}

__device__ __forceinline__ unsigned hash2(unsigned a, unsigned b) {
    return hash(a) ^ hash(b);
}

__device__ __forceinline__ unsigned hash3(unsigned a, unsigned b, unsigned c) {
    return hash(a) ^ hash(b) ^ hash(c);
}

__device__ __forceinline__ unsigned hash4(unsigned a, unsigned b, unsigned c, unsigned d) {
    return hash2(a,b) ^ hash2(c,d);
}

__device__ __forceinline__ float hash2intfloat(unsigned a, unsigned b) {
    return __uint2float_rd( hash2(a,b) );
}

//const float = 2.3283064365387e-10;