#include <stdio.h>
#include <iostream>

__global__
void saxpy(int n, float a, float *x, float *y){
          // n length, a scalar, x and y array pointers

  // indexing like grid(1)
  int idx = blockIdx.x + blockDim.x + threadIdx.x;
  if (idx<n)
    y[idx] = a * x[idx] + y[idx];
  // printf("%d ",idx);
}

// main
int main(void){


  // Parameters
  int N = 100;//<<20; // 1*2^20
  float a = 2.0f;
  int bsize = 256;
  int gsize = (N+bsize-1)/bsize;


  // declare array pointers in host and device
  float *x, *y, *dev_x, *dev_y;

  // host allocate
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  // allocate in device // in unified memory, we would use cudaMallocManaged()
  // at addres of dev_x, size
  cudaMalloc(&dev_x, N*sizeof(float));
  cudaMalloc(&dev_y, N*sizeof(float));

  // populate host arrays
  for (int idx=0; idx<N; idx++){
    x[idx] = 1.0f;
    y[idx] = 2.0f;    
  }

  // copy to device memory
  // https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g48efa06b81cc031b2aa6fdc2e9930741.html
  // destination, source, byteCount, typeTransfer
  cudaError_t err = cudaMemcpy(dev_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err!=cudaSuccess){
    std::cerr<<"cudaMemError: "<<cudaGetErrorString(err)<<std::endl;
    return -1;
  }
  err = cudaMemcpy(dev_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
    std::cerr<<"cudaMemError: "<< cudaGetErrorString(err)<< std::endl;
    return -1;
  }


  // we have run error checks after each device-memory assignation


  // now run the SAXPY kernel
  // num elem, alpha scalar, device array 1, device array 2
  saxpy<<<gsize, bsize>>>(N,a, dev_x, dev_y);
  err = cudaGetLastError();
  if (err != cudaSuccess){
    std::cerr<<"Kernel launch err: "<<cudaGetErrorString(err)<<std::endl;
    return -1;
  }


  // numerical error checks
  float maxerr = 0.0f;
  for (int idx = 0; idx<N; idx++){
    maxerr = maxerr +fabs(y[idx]-4.0f);//max(maxerr, fabs(y[idx]-4.0f));
  }
  printf("Max err: %f\n", maxerr);



  cudaFree(dev_x);
  cudaFree(dev_y);
  free(x);
  free(y);


}
