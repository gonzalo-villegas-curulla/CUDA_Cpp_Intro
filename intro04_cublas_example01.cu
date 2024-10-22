#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
//#include "cublas_utils.h"



int main(){

    int N = 1 << 20; // this mean 1*2^20 (ca. 1 M)

    // call cublas
    //cublasInit(); // deprecated
    cublasHandle_t handle;
    cublasCreate(&handle);

    //cublasDestroy(handle);


return cudaSuccess;
}
