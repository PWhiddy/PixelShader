#define GLM_FORCE_CUDA
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
//#include "noise.h"
//#include "cuda_noise.h"

__device__ __forceinline__ float hash1( uint n ) 
{
    // integer hash copied from Hugo Elias
	n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float( n & 0x7fffffffU)/float(0x7fffffff);
}

__device__ __forceinline__ glm::vec3 hash33(glm::vec3 p3)
{
	p3 = glm::fract(p3 * glm::vec3(.1031f,.11369f,.13787f));
    p3 += glm::dot(p3, glm::vec3(p3.y, p3.x, p3.z)+19.19f);
    return -1.0f + 2.0f * glm::fract(glm::vec3((p3.x + p3.y)*p3.z, (p3.x+p3.z)*p3.y, (p3.y+p3.z)*p3.x));
}

__device__ __forceinline__ float hash31(glm::vec3 p3)
{
	p3  = fract(p3 * glm::vec3(.1031,.11369,.13787));
    p3 += glm::dot(p3, glm::vec3(p3.y,p3.z,p3.x) + 19.19f);
    return -1.0f + 2.0f * glm::fract((p3.x + p3.y) * p3.z);
}

__device__ __forceinline__ float hash71( glm::vec3 p, glm::vec3 dir, int t) {
    float a = hash1( uint(t) );
 	float b = hash31(p);
    float c = hash31(dir);
    return hash31(glm::vec3(a,b,c));
}

// from https://math.stackexchange.com/questions/44689/how-to-find-a-random-axis-or-unit-vector-in-3d
__device__ __forceinline__ glm::vec3 randomDir( glm::vec3 p, glm::vec3 dir, int t) {
    float a = hash1( uint(t) );
 	float b = hash31(p);
    float c = hash31(dir);
    float theta = 6.2831853f*hash31(glm::vec3(a,b,c));
    float z = 2.0f*hash31( 
        glm::vec3( c+1.0f, 2.0f*a+3.5f, b*1.56f+9.0f ) ) - 1.0f;
    float m = sqrt(1.0f-z*z);
   	return glm::vec3( m*sin(theta), m*cos(theta), z );
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
    sum += 0.023125f*simplex_noise(32.0f*p);
    sum += 0.015625f*simplex_noise(64.0f*p);
    sum += 0.0078125*simplex_noise(128.0f*p);
    sum += 0.00390625f*simplex_noise(256.0f*p);
    sum += 0.001953125f*simplex_noise(512.0f*p);
    sum += 0.0009765625f*simplex_noise(1024.0f*p);
    sum += 0.00048828125f*simplex_noise(2048.0f*p);
    return sum * 0.5f + 0.5f;
}

__device__ __forceinline__ float fractal_noiseRough(glm::vec3 p) {
    float sum = 0.0f;
    sum += simplex_noise(p);
    sum += 2.7f*simplex_noise(2.0f*p);
    sum += 2.55f*simplex_noise(4.0f*p);
    sum += 1.885f*simplex_noise(8.0f*p);
    sum += 0.8225f*simplex_noise(16.0f*p);
    sum += 0.43125f*simplex_noise(32.0f*p);
    sum += 0.225625f*simplex_noise(64.0f*p);
    sum += 0.1078125*simplex_noise(128.0f*p);
    sum += 0.02690625f*simplex_noise(256.0f*p);
    return sum * 0.5f + 0.5f;
}

__device__  __forceinline__ glm::vec2 rotate(glm::vec2 p, float a)
{
    return glm::vec2(p.x*cos(a) - p.y*sin(a),
                     p.y*cos(a) + p.x*sin(a));
}

__device__ __forceinline__ float sdSphere(glm::vec3 p, float r) {
    return glm::length(p)-r;
}

__device__ __forceinline__ float mushSphere(glm::vec3 p, float t) {
    p.z -= 0.8f;
    return sdSphere(p, 0.8)/* + 
          (0.2f*sin(t*0.02f)+0.215f)*0.23f * 
          fractal_noiseRough(
              glm::vec3(0.12f*p.x,0.12f*p.y,0.12*p.z) + 
              50.0f + 
              glm::vec3(0.0f,0.0f,0.0001f*t)
            )*/;
}

__device__ __forceinline__ float sdBox( glm::vec3 p, glm::vec3 b )
{
    p.y -= 1.5f;
    glm::vec3 d = glm::abs(p) - b;
    return glm::min(glm::max(d.x,glm::max(d.y,d.z)),0.0f) + glm::length(glm::max(d,glm::vec3(0.0f)));
}

__device__ __forceinline__ float boxDist(glm::vec3 p, float t) {
    return -sdBox(p, glm::vec3(3.5,2.5,4.5));//+0.02*fractal_noiseRough(0.15f*p+30.0f);
}

__device__  float map(glm::vec3 p, float t) {
    float d;
    d = mushSphere(p, t);
    d = fminf(boxDist(p, t), d);
    return d;
}

__device__  glm::vec3 calcNormal( glm::vec3 pos, float t )
{
    glm::vec2 e = glm::vec2(1.0,-1.0)*0.5773f*0.0005f;
    return glm::normalize( glm::vec3(e.x,e.y,e.y)*map( pos + glm::vec3(e.x,e.y,e.y), t ) + 
					  glm::vec3(e.y,e.y,e.x)*map( pos + glm::vec3(e.y,e.y,e.x), t ) + 
					  glm::vec3(e.y,e.x,e.y)*map( pos + glm::vec3(e.y,e.x,e.y), t ) + 
					  glm::vec3(e.x,e.x,e.x)*map( pos + glm::vec3(e.x,e.x,e.x), t ) );
}

__device__  glm::vec3 intersect(glm::vec3 ray_pos, glm::vec3 ray_dir, float t)
{
    for (int i=0; i<128; i++) {
        float dist = map(ray_pos, t);
        if (dist < 0.002 || dist > 100.0) break;
        ray_pos += dist * ray_dir * 0.9f;
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

    //glm::vec3 light_dir = glm::normalize(glm::vec3(0.1, 1.0, -0.5));
    float light_height = 2.7f;

    const int aa_size = 64;
    const int sample_count = aa_size*aa_size;
    const float aa_inv = 1.0f/float(aa_size);

    glm::vec3 final_color = glm::vec3(0.0f, 0.0f, 0.0f);

    for (int sample_index = 0; sample_index<sample_count; sample_index++) {

        float aa_x = float(sample_index % 16)*aa_inv;
        float aa_y = float(sample_index / 16)*aa_inv;

        // Create Normalized UV image coordinates
        float uvx =  (float(x)+aa_x)/float(x_dim)-0.5;
        float uvy = -(float(y)+aa_y)/float(y_dim)+0.5;
        uvx *= float(x_dim)/float(y_dim);     

        // Set up ray originating from camera
        glm::vec3 ray_pos = glm::vec3(0.0, 0.0, -1.5);
        glm::vec2 pos_rot = rotate(glm::vec2(ray_pos.x, ray_pos.z), 0.0);
        ray_pos.x = pos_rot.x;
        ray_pos.z = pos_rot.y;
        glm::vec3 ray_dir = glm::normalize(glm::vec3(uvx,uvy,0.5));
        glm::vec2 dir_rot = rotate(glm::vec2(ray_dir.x, ray_dir.z), 0.0);
        ray_dir.x = dir_rot.x;
        ray_dir.z = dir_rot.y;

        glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 incoming = glm::vec3(0.0f, 0.0f, 0.0f);

        const int max_bounces = 6;
        
        for (int bounce = 0; bounce < max_bounces; bounce++)
        {
            ray_pos = intersect(ray_pos, ray_dir, time);
            if (mushSphere(ray_pos, time) < 0.1f) {
                color *= glm::vec3(1.0, 0.8, 0.6);
                incoming += 0.02f;
            } else if (ray_pos.y > light_height) {
                incoming += 1.6f;
                break;
            } else {
                color *= glm::vec3(1.0f, 0.2f, 1.0f);
                incoming += 0.01f;
            }

            glm::vec3 normal = calcNormal(ray_pos, time);

            float rand = hash71(ray_pos, ray_dir, sample_index);
            float shiny = fmod(4.0f*(ray_pos.x+ray_pos.y+ray_pos.z), 1.0);
            if (rand > shiny*shiny) {
                // specular reflection
                ray_dir = glm::reflect(ray_dir, normal);
            } else {
                // diffuse scatter
                glm::vec3 ndir = randomDir( ray_pos, ray_dir, sample_index+10 );
                ray_dir = glm::normalize(8.0f*(ndir+normal*1.002f));
            }
            ray_pos += 0.01f*ray_dir;

        }

        final_color += incoming*color;
        
        /*
        glm::vec3 background = glm::vec3(0.87);
        glm::vec3 normal = calcNormal(ray_pos, time);
        float value = glm::dot(normal,light_dir);
        color = glm::vec3(value, value, value);
        if (boxDist(ray_pos, time) < 0.05) {
            color.y -= 0.4;
            color.z -= 0.5;
        }
        if (glm::length(ray_pos) > 50.0) color = background;
        //color = make_float3(rng::simplexNoise(make_float3(uvx*25.0,uvy*25.0,0.0)+100.0, 1.0, 123));
        */

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
    }

    final_color /= float(sample_count);

    const int pixel = 3*((y-y_offset)*x_dim+x);
    image[pixel+0] = (uint8_t)(fmin(255.0f*final_color.x, 255.0f));
    image[pixel+1] = (uint8_t)(fmin(255.0f*final_color.y, 255.0f));
    image[pixel+2] = (uint8_t)(fmin(255.0f*final_color.z, 255.0f));
}
