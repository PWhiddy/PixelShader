#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "render_pixel.cu"

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

    for (int time_step = 0; time_step < t_dim; time_step++) {
        
        std::cout << "Step " << time_step+1 << "\n";

        int num_devices = 0;
        cudaGetDeviceCount( &num_devices );
        #pragma omp parallel num_threads( num_devices )
        {
            int dev_id = omp_get_thread_num();
            cudaSetDevice( dev_id );

            float measured_time=0.0f;
            cudaEvent_t start, stop;
            cudaEventCreate( &start );
            cudaEventCreate( &stop  );
            cudaEventRecord( start, 0 );

            int y_offset = dev_id*(y_dim/num_devices);
            dim3 block_size(32,16,1);
            dim3 image_grid(
                (x_dim+block_size.x-1)/block_size.x, 
                ((y_dim+block_size.y-1)/block_size.y)/num_devices, 1);
        
            int img_bytes = (3 * sizeof(uint8_t) * x_dim * y_dim)/num_devices;
            uint8_t *device_img;
            cudaMalloc( (void**) &device_img, img_bytes );

            render_pixel<<<image_grid, block_size>>>(
                device_img,
                x_dim,
                y_dim,
                time_step,
                y_offset
            );
    
            cudaMemcpy( 
                img+(3*y_offset*x_dim), 
                device_img, 
                img_bytes, 
                cudaMemcpyDeviceToHost
            );

            cudaFree(device_img);

            cudaEventRecord( stop, 0 );
            cudaThreadSynchronize();
            cudaEventElapsedTime( &measured_time, start, stop );
    
            cudaEventDestroy( start );
            cudaEventDestroy( stop );
    
            std::cout << "GPU: " << dev_id <<  " Render Time: " << measured_time << "ms\n";

        }


        save_image(
            img, 
            x_dim, 
            y_dim, 
            "output/F" + pad_number(time_step+1) + ".ppm"
        );
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ));
    cudaThreadExit();
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
