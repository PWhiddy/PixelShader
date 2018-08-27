#define GLM_FORCE_CUDA
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
//#include "noise.h"
//#include "cuda_noise.h"



__device__ __forceinline__ glm::vec3 hash33(glm::vec3 p3)
{
	p3 = glm::fract(p3 * glm::vec3(.1031f,.11369f,.13787f));
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.x, p3.z)+19.19f);
    return -1.0f + 2.0f * glm::fract(glm::vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

__device__ __forceinline__ float simplex_noise(glm::vec3 p)
{
    const float K1 = 0.333333333f;
    const float K2 = 0.166666667f;
    
    glm::vec3 i = glm::floor(p + (p.x + p.y + p.z) * K1);
    glm::vec3 d0 = p - (i - (i.x + i.y + i.z) * K2);
    
    // thx nikita: https://www.shadertoy.com/view/XsX3zB
    glm::vec3 e = glm::step(glm::vec3(0.0), d0 - glm::vec3(d0.y, d0.z, d0.x));
	glm::vec3 i1 = e * (1.0f - glm::vec3(e.z, e.x, e.y));
	glm::vec3 i2 = 1.0f - glm::vec3(e.z,e.x,e.y) * (1.0f - e);
    
    glm::vec3 d1 = d0 - (i1 - 1.0f * K2);
    glm::vec3 d2 = d0 - (i2 - 2.0f * K2);
    glm::vec3 d3 = d0 - (1.0f - 3.0f * K2);
    
    glm::vec4 h = glm::max(0.6f - glm::vec4(glm::dot(d0, d0), glm::dot(d1, d1), glm::dot(d2, d2), glm::dot(d3, d3)), 0.0f);
    glm::vec4 n = h * h * h * h * glm::vec4(glm::dot(d0, hash33(i)), glm::dot(d1, hash33(i + i1)), glm::dot(d2, hash33(i + i2)), glm::dot(d3, hash33(i + 1.0f)));
    
    return glm::dot(glm::vec4(31.316f), n);
}

__device__ __forceinline__ float fractal_noise(glm::vec3 p) {
    float sum = 0.0f;
    sum += simplex_noise(p);
    sum += 0.5f*simplex_noise(2.0f*p);
    sum += 0.25f*simplex_noise(4.0f*p);
    sum += 0.125f*simplex_noise(8.0f*p);
    sum += 0.0625f*simplex_noise(16.0f*p);
    sum += 0.03125f*simplex_noise(32.0f*p);
    sum += 0.015625f*simplex_noise(64.0f*p);
    sum += 0.0078125*simplex_noise(128.0f*p);
    sum += 0.00390625f*simplex_noise(256.0f*p);
    sum += 0.001953125f*simplex_noise(512.0f*p);
    sum += 0.0009765625f*simplex_noise(1024.0f*p);
    sum += 0.00048828125f*simplex_noise(2048.0f*p);
    return sum * 0.5f + 0.5f;
}


__device__ glm::vec2 rotate(glm::vec2 p, float a)
{
    return glm::vec2(p.x*cos(a) - p.y*sin(a),
                     p.y*cos(a) + p.x*sin(a));
}

__device__ float sdSphere(glm::vec3 p, float r) {
    return glm::length(p)-r;
}

__device__ float sdBox( glm::vec3 p, glm::vec3 b )
{
    glm::vec3 d = glm::abs(p) - b;
      return glm::min(glm::max(d.x,glm::max(d.y,d.z)),0.0f) + glm::length(glm::max(d,glm::vec3(0.0f)));
}

/*
__device__ float fractalNoise(float3 p) {
    p += 300.0f;
    float result = 0.0f;
    result += rng::simplexNoise(p*1.0, 1.0, 123) * 1.0f;
    result += rng::simplexNoise(p*2.0, 1.0, 123) * 0.5f;
    result += rng::simplexNoise(p*4.0, 1.0, 123) * 0.25f;
    result += rng::simplexNoise(p*8.0, 1.0, 123) * 0.125f;
    result += rng::simplexNoise(p*16.0, 1.0, 123) * 0.0625f;
    result += rng::simplexNoise(p*32.0, 1.0, 123) * 0.03125f;
    result += rng::simplexNoise(p*64.0, 1.0, 123) * 0.015625f;
    return result;
}
*/

__device__ float map(glm::vec3 p, float t) {
    float d;
    d =  sdSphere(p, 0.8)+(sin(t*0.1f))*0.73f*fractal_noise(glm::vec3(0.12f*p.x,0.12f*p.y,0.008f*t)+50.0f+t*0.01f*t);
    //d = fminf(-sdBox(p, make_float3(2.0,2.0,2.0)), d);
    return d;
}

__device__ glm::vec3 calcNormal( glm::vec3 pos, float t )
{
    glm::vec2 e = glm::vec2(1.0,-1.0)*0.5773f*0.0005f;
    return glm::normalize( glm::vec3(e.x,e.y,e.y)*map( pos + glm::vec3(e.x,e.y,e.y), t ) + 
					  glm::vec3(e.y,e.y,e.x)*map( pos + glm::vec3(e.y,e.y,e.x), t ) + 
					  glm::vec3(e.y,e.x,e.y)*map( pos + glm::vec3(e.y,e.x,e.y), t ) + 
					  glm::vec3(e.x,e.x,e.x)*map( pos + glm::vec3(e.x,e.x,e.x), t ) );
}

__device__ glm::vec3 intersect(glm::vec3 ray_pos, glm::vec3 ray_dir, float t)
{
    for (int i=0; i<1024; i++) {
        float dist = map(ray_pos, t);
        if (dist < 0.002 || dist > 100.0) break;
        ray_pos += dist * ray_dir * 0.15f;
    }
    return ray_pos;
}

__global__ void render_pixel ( 
    uint8_t *image,
    int x_dim,
    int y_dim,
    int time_step,
    int y_offset
){
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y+y_offset;
    if (x >= x_dim || y >= y_dim) return;

    float time = float(time_step);
    
    // Create Normalized UV image coordinates
    float uvx =  float(x)/float(x_dim)-0.5;
    float uvy = -float(y)/float(y_dim)+0.5;
    uvx *= float(x_dim)/float(y_dim);     

    glm::vec3 light_dir = glm::normalize(glm::vec3(0.1, 1.0, -0.5));

    // Set up ray originating from camera
    glm::vec3 ray_pos = glm::vec3(0.0, 0.0, -1.5);
    glm::vec2 pos_rot = rotate(glm::vec2(ray_pos.x, ray_pos.z), 0.0);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    glm::vec3 ray_dir = glm::normalize(glm::vec3(uvx,uvy,0.5));
    glm::vec2 dir_rot = rotate(glm::vec2(ray_dir.x, ray_dir.z), 0.0);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;

    glm::vec3 color = glm::vec3(0.95,0.92,0.96);
    const int max_bounces = 6;
    
    //for (int bounce = 0; bounce < max_bounces; bounce++)
    //{
        ray_pos = intersect(ray_pos, ray_dir, time);

    //}
    
    
    glm::vec3 background = glm::vec3(0.87);
    glm::vec3 normal = calcNormal(ray_pos, time);
    float value = glm::dot(normal,light_dir);
    color = glm::vec3(value, value, value);
    if (glm::length(ray_pos) > 10.0) color = background;
    //color = make_float3(rng::simplexNoise(make_float3(uvx*25.0,uvy*25.0,0.0)+100.0, 1.0, 123));

    /*
    const float3 dir_to_light = normalize(light_dir);
    const float occ_thresh = 0.001;
    float d_accum = 1.0;
    float light_accum = 0.0;
    float temp_accum = 0.0;
    

    // Trace ray through volume
    for (int step=0; step<512; step++) {
        // At each step, cast occlusion ray towards light source
        float c_density = get_cellF(ray_pos, vd, volume);
        float3 occ_pos = ray_pos;
        ray_pos += ray_dir*step_size;
        // Don't bother with occlusion ray if theres nothing there
        if (c_density < occ_thresh) continue;
        float transparency = 1.0;
        for (int occ=0; occ<512; occ++) {
            transparency *= fmax(1.0-get_cellF(occ_pos, vd, volume),0.0);
            if (transparency < occ_thresh) break;
            occ_pos += dir_to_light*step_size;
        }
        d_accum *= fmax(1.0-c_density,0.0);
        light_accum += d_accum*c_density*transparency;
        if (d_accum < occ_thresh) break;
    }
    
    // gamma correction
    light_accum = pow(light_accum, 0.45);
    */
    
    //const float conv_range = 2.3283064365387e-10;
    /*
    float val = hash2intfloat(x,y)*conv_range;
    color.x = val;
    color.y = val;
    color.z = val;
    */

    //float4 rand = randomInt44( make_uint4( x, y, 5, time_step ) );
    //color.x = __uint2float_rd(rand.x) * conv_range;
    //color.y = __uint2float_rd(rand.y) * conv_range;
    //color.z = __uint2float_rd(rand.z) * conv_range;
    
    //float val = fractal4( make_float4( float(x)*0.002f, float(y)*0.002f, 6.0f, float(time_step)*0.07f ) );
    //float val = 0.5;
    //glm::vec3 position = glm::vec3(float(x)*0.01f, float(y)*0.01f, 1.0f);
    //val = fractal_noise(position);
    //val = 0.5f*simplex_noise(position) + 0.5f;

    const int pixel = 3*((y-y_offset)*x_dim+x);
    image[pixel+0] = (uint8_t)(fmin(255.0f*color.x, 255.0f));
    image[pixel+1] = (uint8_t)(fmin(255.0f*color.y, 255.0f));
    image[pixel+2] = (uint8_t)(fmin(255.0f*color.z, 255.0f));
}
