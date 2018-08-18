#include "cutil_math.h"

//#define MOD3 make_float3(.1031,.11369,.13787)
//#define MOD3 float3(443.8975,397.2973, 491.1871)
const float3 MOD3 = make_float3(.1031,.11369,.13787);

__device__ float hash31(float3 p3)
{
	p3  = fract(p3 * MOD3);
    p3 += dot(p3, make_float3(p3.y,p3.z,p3.x) + 19.19);
    return -1.0f + 2.0f * fract((p3.x + p3.y) * p3.z);
}

__device__ float3 hash33(float3 p3)
{
	p3 = fract(p3 * MOD3);
    p3 += dot(p3, make_float3(p3.y,p3.x,p3.z)+19.19);
    return -1.0f + 2.0f * fract(make_float3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

__device__ float simplex_noise(float3 p)
{
    const float K1 = 0.333333333f;
    const float K2 = 0.166666667f;
    
    float3 i = floor(p + (p.x + p.y + p.z) * K1);
    float3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    float3 e = step(make_float3(0.0), d0 - make_float3(d0.y,d0.z,d0.x));
	float3 i1 = e * (1.0f - make_float3(e.z,e.x,e.y));
	float3 i2 = 1.0f - make_float3(e.z,e.x,e.y) * (1.0f - e);
    
    float3 d1 = d0 - (i1 - 1.0f * K2);
    float3 d2 = d0 - (i2 - 2.0f * K2);
    float3 d3 = d0 - (1.0 - 3.0f * K2);
    
    float4 h = fmaxf(0.6f - make_float4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3)), make_float4(0.0f));
    float4 n = h * h * h * h * make_float4(dot(d0, hash33(i)), dot(d1, hash33(i + i1)), dot(d2, hash33(i + i2)), dot(d3, hash33(i + 1.0)));
    
    return dot(make_float4(31.316), n);
}