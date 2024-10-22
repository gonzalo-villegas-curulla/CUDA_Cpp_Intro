#include <iostream>
#include <math.h>
#include <cstdio>
#include <cstdlib>

__global__ void add(int n, float *x, float*y){
    int idx_init = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x *gridDim.x;

    for (int IDX=idx_init;
            IDX < n;
            IDX += stride)
        y[IDX] = x[IDX] + y[IDX];

}


int main(void){

    int N = 1<<20; // 1M elem
    float *x, *y; // pointer float to x and y arrays

    // unified memory allocation
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize and populate x and y
    for (int idx = 0; idx<N; idx++){
        x[idx] = 1.0f;
        y[idx] = 2.0f;
    }

    // launch kernel
    int bsize = 256;
    int gsize = (N + bsize -1) / bsize;

    add<<<gsize,bsize>>>(N, x, y);

    cudaDeviceSynchronize();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        std::cerr<<"Error synch: "<< cudaGetErrorString(err)<<std::endl;
        return -1;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr<<"Error launching kernel: "<<cudaGetErrorString(err)<<std::endl;
        return -1;
    }


    float maxerr = 0.0f;
    for (int idx=0; idx<N;idx++){
        maxerr = fmax(maxerr, fabs(y[idx]-3.0f));
    }
    std::cout << "Max err: "<< maxerr<< std::endl;

    return 0;

}
