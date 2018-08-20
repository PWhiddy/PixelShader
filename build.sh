#!/bin/sh

CUDA_DIR='usr/local/cuda-9.2'

/${CUDA_DIR}/bin/nvcc create_image.cu -std=c++11 -Xcompiler -fopenmp -I ${CUDA_DIR}/include -L ${CUDA_DIR}/lib64 -lgomp -Wno-deprecated-gpu-targets -o start_render
