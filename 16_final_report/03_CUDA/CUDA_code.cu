//20M19118_CHENYING_CUDA
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
using namespace std;

__global__ void matrix(int N, float *A, float *B, float* C){
    int i = blockIdx.x / N;
    int j = blockIdx.x % N;
    int k = threadIdx.x;
       
    atomicAdd(C+N*i+j, A[N*i+k] * B[N*k+j]);
}


int main(int argc, char** argv) {
  const int N = 256;
  float *A;
  float *B;
  float *C;

  cudaMallocManaged(&A, N*N*sizeof(float));
  cudaMallocManaged(&B, N*N*sizeof(float));
  cudaMallocManaged(&C, N*N*sizeof(float));
  
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
  
  auto tic = chrono::steady_clock::now();

  matrix<<<N*N,N>>>(N,A,B,C);
  cudaDeviceSynchronize();

  auto toc = chrono::steady_clock::now();
  
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);
  double time = chrono::duration<double>(toc-tic).count();
  printf("N    : %d\n",N);
  printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
  printf("error: %lf\n",err/N/N);
}
