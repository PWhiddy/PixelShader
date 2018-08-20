//#include "noise.h"
#include "cuda_noise.h"
#include "cutil_math.h"

__device__ float2 rotate(float2 p, float a)
{
    return make_float2(p.x*cos(a) - p.y*sin(a),
                       p.y*cos(a) + p.x*sin(a));
}

__device__ float sdSphere(float3 p, float r) {
    return length(p)-r;
}

__device__ float sdBox( float3 p, float3 b )
{
      float3 d = fabs(p) - b;
      return fminf(fmaxf(d.x,fmaxf(d.y,d.z)),0.0f) + length(fmaxf(d,make_float3(0.0f)));
}

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

__device__ float map(float3 p) {
    float d;
    d =  sdSphere(p, 0.7)/*+(sin(38.0*p.x)+sin(47.0*p.y)+sin(21.0*p.z))*0.1*/+0.1*fractalNoise(p*2.5);
    d = fminf(-sdBox(p, make_float3(2.0,2.0,2.0)), d);
    return d;
}

__device__ float3 calcNormal( float3 pos )
{
    float2 e = make_float2(1.0,-1.0)*0.5773*0.0005;
    return normalize( make_float3(e.x,e.y,e.y)*map( pos + make_float3(e.x,e.y,e.y) ) + 
					  make_float3(e.y,e.y,e.x)*map( pos + make_float3(e.y,e.y,e.x) ) + 
					  make_float3(e.y,e.x,e.y)*map( pos + make_float3(e.y,e.x,e.y) ) + 
					  make_float3(e.x,e.x,e.x)*map( pos + make_float3(e.x,e.x,e.x) ) );
}

__global__ void render_pixel ( 
    uint8_t *image, 
    int x_dim, 
    int y_dim, 
    int time_step
){
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    if (x >= x_dim || y >= y_dim) return;

    float time = float(time_step);
    
    // Create Normalized UV image coordinates
    float uvx =  float(x)/float(x_dim)-0.5;
    float uvy = -float(y)/float(y_dim)+0.5;
    uvx *= float(x_dim)/float(y_dim);     

    float3 light_dir = normalize(make_float3(0.1, 1.0, -0.5));

    // Set up ray originating from camera
    float3 ray_pos = make_float3(0.0, 0.0, -1.5);
    float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), 0.0);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
    float2 dir_rot = rotate(make_float2(ray_dir.x, ray_dir.z), 0.0);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;

    for (int i=0; i<1024; i++) {
        float dist = map(ray_pos);
        if (dist < 0.002 || dist > 100.0) break;
        ray_pos += dist * ray_dir * 0.15;
    }

    float3 background = make_float3(0.87);
    float3 normal = calcNormal(ray_pos);
    float value = dot(normal,light_dir);
    float3 color = make_float3(value, value, value);
    if (length(ray_pos) > 10.0) color = background;
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
    const int pixel = 3*(y*x_dim+x);
    image[pixel+0] = (uint8_t)(fmin(255.0*color.x, 255.0));
    image[pixel+1] = (uint8_t)(fmin(255.0*color.y, 255.0));
    image[pixel+2] = (uint8_t)(fmin(255.0*color.z, 255.0));
}
