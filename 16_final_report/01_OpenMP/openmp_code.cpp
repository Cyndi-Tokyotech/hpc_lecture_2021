//20M19118 CHENYING_01_OPENMP

// Compile with openmp_code.cpp -fopenmp -std=c++11

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv)
{
  const int N = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
    }
  }
// Record starttime
  auto tic = chrono::steady_clock::now();

#pragma omp parallel for shared(C)
  for (int i=0; i<N; i++)
#pragma omp parallel for shared(C)
    for (int j=0; j<N; j++)
#pragma omp parallel for shared(C)
      for (int k=0; k<N; k++)
        C[N*i+j] += A[N*i+k] * B[N*k+j];
    
// Record endtime
  auto toc = chrono::steady_clock::now();
// Calculate time use
  double time = chrono::duration<double>(toc-tic).count();

// Record Error
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      for (int k=0; k<N; k++)
        C[N*i+j] -= A[N*i+k] * B[N*k+j];
  double err = 0;
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      err += fabs(C[N*i+j]);

  printf("N    : %d\n",N);
  printf("total: %lf s (%lf GFlops)\n",time,2.*N*N*N/time/1e9);
  printf("error: %lf\n",err/N/N);
}
