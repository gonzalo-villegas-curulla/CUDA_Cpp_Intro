#include <cstdio>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>


__global__

void saxpy(int n, float a, float * x, float* y){
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx<n){
    //    y[idx] = a*x[idx] + y[idx];
    //}

    // which can be expressed better:

    for (int idx = blockDim.x * blockIdx.x + threadIdx.x ;
            idx < n;
            idx += blockDim.x * gridDim.x) // increase by stride
    {
        y[idx] = a * x[idx] + y[idx];
    }
}



int main(void){

    int SMs; // stream multiprocessors
    //char cudaDevAttrMultiProcessorCount;
    //cudaDeviceGetAttribute(&SMs,0,0);
    int devId;
    cudaDeviceGetAttribute(&SMs, cudaDevAttrMaxGridDimX, &devId);



    int N = 10000;
    float a=3.0, val1=1.0, val2=2.0;

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // populate

    for (int jdx=0; jdx<N; jdx++){
        x[jdx] = val1;
        y[jdx] = val2;
    }

    // operate saxpy
    saxpy<<<4096,256>>>(N, a, x, y);
    // error check
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        std::cerr << "Cuda synch err: "<< cudaGetErrorString(err) << std::endl;
        return -1;
    }
    err = cudaGetLastError();
    if (err!=cudaSuccess){
        std::cerr << "Cuda kernel launch error: "<<cudaGetErrorString(err)<< std::endl;
        return -1;
    }

    // print out values
    //for (int idx=0; idx<N; idx++){
    //    printf("%1.2f ",y[idx]);
    //}

    float maxerr = 0.0f;
    float total = a*val1 + val2;
    for (int idx=0; idx<N;idx++)
    {
        maxerr = fmax(maxerr, fabs(y[idx]-total));
    }
    std::cout << "Max err: "<< maxerr<<std::endl;

    return EXIT_SUCCESS;
}

