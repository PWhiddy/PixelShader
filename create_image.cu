#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include "trace_pixel.cu"

void render_images(int x_dim, int y_dim, int t_dim);
void save_image(uint8_t *pixels, int x_dim, int y_dim, std::string name);
std::string pad_number(int n);

int main(int argc, char* args[])
{
    if (argc == 4)
    {
        std::istringstream x_string( args[1] );
        std::istringstream y_string( args[2] );
        std::istringstream t_string( args[3] );
        int x_dim;
        int y_dim;
        int t_dim;

        if  ( 
             x_string >> x_dim && 
             y_string >> y_dim &&
             t_string >> t_dim
            ) 
        {
            render_images(x_dim,y_dim,t_dim);
        } else {
            std::cout << "Error parsing dimensions\n";
        }
    } else {
        std::cout << "Incorrect number of arguments provided\n";
        std::cout << "Requires: X dimension, Y dimension, T dimension\n";
    }

    return 0;
}

void render_images(int x_dim, int y_dim, int t_dim) {

    uint8_t *img = new uint8_t[3*x_dim*y_dim];

    dim3 block_size(32,32,1);
    dim3 image_grid(
        (x_dim+block_size.x-1)/block_size.x, 
        (y_dim+block_size.y-1)/block_size.y, 1);

    int img_bytes = 3 * sizeof(uint8_t) * x_dim * y_dim;
    uint8_t *device_img;
    cudaMalloc( (void**) &device_img, img_bytes );

    for (int time_step = 0; time_step < t_dim; time_step++) {
        
        std::cout << "Step " << time_step+1 << "\n";

        float measured_time=0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate( &start );
        cudaEventCreate( &stop  );
        cudaEventRecord( start, 0 );

        render_pixel<<<image_grid, block_size>>>(
            device_img,
            x_dim,
            y_dim,
            time_step
        );

        cudaMemcpy( 
            img, 
            device_img, 
            img_bytes, 
            cudaMemcpyDeviceToHost
        );

        cudaEventRecord( stop, 0 );
        cudaThreadSynchronize();
        cudaEventElapsedTime( &measured_time, start, stop );

        cudaEventDestroy( start );
        cudaEventDestroy( stop );

        std::cout << "Render Time: " << measured_time << "\n";

        save_image(
            img, 
            x_dim, 
            y_dim, 
            "output/F" + pad_number(f+1) + ".ppm"
        );
    }

    cudaFree(device_img);
    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ));
    cudaThreadExit();

    return 0;
}

std::string pad_number(int n)
{
    std::ostringstream ss;
    ss << std::setw( 7 ) << std::setfill( '0' ) << n;
    return ss.str();
}

void save_image(uint8_t *pixels, int x_dim, int y_dim, std::string name) {
    std::ofstream file(name, std::ofstream::binary);
    if (file.is_open()) {
        file << "P6\n" 
             << x_dim
            << " " 
            << y_dim
            << "\n" 
            << "255\n";
        file.write((char *)pixels, x_dim*y_dim*3);
        file.close();
    } else {
        std::cout << "Could not open file :(\n";
    }
}