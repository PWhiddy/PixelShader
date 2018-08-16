#include "cutil_math.h"

__device__ float2 rotate(float2 p, float a)
{
    return make_float2(p.x*cos(a) - p.y*sin(a),
                       p.y*cos(a) + p.x*sin(a));
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

    // Set up ray originating from camera
    float3 ray_pos = make_float3(0.0, 0.0, 0.0);
    float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), 0.0);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    ray_pos += v_center;
    float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
    float2 dir_rot = rotate(make_float2(ray_dir.x, ray_dir.z), 0.0);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;
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
    image[pixel+0] = (uint8_t)(fmin(255.0*uvx, 255.0));
    image[pixel+1] = (uint8_t)(fmin(255.0*uvy, 255.0));
    image[pixel+2] = (uint8_t)(fmin(255.0*(0.5*sin(uvx)+0.5), 255.0));
}