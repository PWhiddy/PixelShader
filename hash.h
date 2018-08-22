#include "cutil_math.h"
/*
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
*/

// https://www.shadertoy.com/view/4dlcR4
__device__ __forceinline__ uint hash2(uint x, uint y)
{
  x *= 0x3504f335u;   // 15 | 101 | 41*79*274627
  y *= 0x8fc1ecd5u;   // 18 | 101 | 5*482370193
  x ^= y;    // combine
  x *= 741103597u;     // 18 | 101 | 5*482370193
  return x;
}

// https://www.shadertoy.com/view/XlXcW4
__device__ __forceinline__ uint3 hash33( uint3 x )
{
    /*
    x = ((x>>8U)^x.yzx)*1103515245U;
    x = ((x>>8U)^x.yzx)*1103515245U;
    x = ((x>>8U)^x.yzx)*1103515245U;
    //return vec3(x)*(1.0/float(0xffffffffU));
    */
    uint temp_x = x.x;
    x.x = ((x.x>>8U)^x.y)*1103515245U;
    x.y = ((x.y>>8U)^x.z)*1103515245U;
    x.z = ((x.z>>8U)^temp_x)*1103515245U;

    temp_x = x.x;
    x.x = ((x.x>>8U)^x.y)*1103515245U;
    x.y = ((x.y>>8U)^x.z)*1103515245U;
    x.z = ((x.z>>8U)^temp_x)*1103515245U;

    temp_x = x.x;
    x.x = ((x.x>>8U)^x.y)*1103515245U;
    x.y = ((x.y>>8U)^x.z)*1103515245U;
    x.z = ((x.z>>8U)^temp_x)*1103515245U;

    return x;
}

__device__ __forceinline__ float hash2intfloat(uint a, uint b) {
    return __uint2float_rd( hash2(a,b) );
}

//const float = 2.3283064365387e-10;