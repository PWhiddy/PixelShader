#include "hash.h"

__device__ __forceinline__ float noise4( float4 p ) {

    float4 lower = floor(p);
    float4 disp = p-lower;
    disp = disp*disp*(3.0-2.0*disp);
    uint4 low = make_uint4(
        __float2uint_rd(lower.x),
        __float2uint_rd(lower.y),
        __float2uint_rd(lower.z),
        __float2uint_rd(lower.w)
    );

    float sum = 0.0;

    #pragma unroll
    for (int x=0; x<=1; x++)
    {
        #pragma unroll
        for (int y=0; y<=1; y++)
        {
            #pragma unroll
            for (int z=0; z<=1; z++)
            {
                #pragma unroll
                for (int w=0; w<=1; w++)
                {
                    sum += abs(float(1-x)-disp.x) *
                           abs(float(1-y)-disp.y) *
                           abs(float(1-z)-disp.z) *
                           abs(float(1-w)-disp.w) * 
                           dot(randomInt44( make_uint4( low.x+x, low.y+y, low.z+z, low.w+w ) )*2.0f-1.0f, disp);
                }
            }
        }
    }
    return sum;
}

/*
// From https://nullprogram.com/blog/2018/07/31/
__device__ __forceinline__ unsigned int hash(unsigned int a)
{
    a ^= a >> 16;
    a *= 0x7feb352d;
    a ^= a >> 15;
    a *= 0x846ca68b;
    a ^= a >> 16;
    return a;
}

__device__ __forceinline__ unsigned int hash2(unsigned int a, unsigned int b) {
    return hash(a) ^ hash(b);
}

__device__ __forceinline__ unsigned int hash3(unsigned int a, unsigned int b, unsigned int c) {
    return hash(a) ^ hash(b) ^ hash(c);
}

__device__ __forceinline__ unsigned int hash4(unsigned int a, unsigned int b, unsigned int c unsigned int d) {
    return hash2(a,b) ^ hash2(c,d);
}

__device__ __forceinline__ unsigned int hash2intfloat(unsigned int a, unsigned int b) {
    return __uint2float_rd( hash2(a, b) );
}
*/
/*
__device__ float hash31(float3 p3)
{
	p3  = fract(p3 * make_float3(.1031f,.11369f,.13787f));
    p3 += dot(p3, make_float3(p3.y,p3.z,p3.x) + 19.19f);
    return -1.0f + 2.0f * fract((p3.x + p3.y) * p3.z);
}

__device__ float3 hash33(float3 p3)
{
	p3 = fract(p3 * make_float3(.1031f,.11369f,.13787f));
    p3 += dot(p3, make_float3(p3.y,p3.x,p3.z)+19.19f);
    return -1.0f + 2.0f * fract(make_float3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

__device__ float simplex_noise(float3 p)
{
    const float K1 = 0.333333333f;
    const float K2 = 0.166666667f;
    
    float3 i = floor(p + (p.x + p.y + p.z) * K1);
    float3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    float3 e = step(make_float3(0.0f), d0 - make_float3(d0.y,d0.z,d0.x));
	float3 i1 = e * (1.0f - make_float3(e.z,e.x,e.y));
	float3 i2 = 1.0f - make_float3(e.z,e.x,e.y) * (1.0f - e);
    
    float3 d1 = d0 - (i1 - 1.0f * K2);
    float3 d2 = d0 - (i2 - 2.0f * K2);
    float3 d3 = d0 - (1.0 - 3.0f * K2);
    
    float4 h = fmaxf(0.6f - make_float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), make_float4(0.0f));
    float4 n = h * h * h * h * make_float4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0f)));
    
    return dot(make_float4(31.316f), n);
}
*/