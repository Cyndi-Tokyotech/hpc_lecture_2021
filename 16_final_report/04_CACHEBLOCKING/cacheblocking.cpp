//20M19118_CHENYING_cacheblocking
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv) {
  const int N = 256;
  vector<float> A(N*N);
  vector<float> B(N*N);
  vector<float> C(N*N);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[N*i+j] = drand48();
      B[N*i+j] = drand48();
      C[N*i+j] = 0;
    }
  }
  
  auto tic = chrono::steady_clock::now();

  const int m = N, n = N, k = N;
  const int kc = 64;
  const int nc = 64;
  const int mc = 64;
  const int nr = 64;
  const int mr = 32;

  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      float Bc[kc*nc];
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p*nc+j] = B[(p+pc)*N+j+jc];
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
    float Ac[mc*kc],Cc[mc*nc];
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = A[(i+ic)*N+p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
                for (int j=jr; j<jr+nr; j++) {
                  Cc[i*nc+j] += Ac[i*kc+kr] * Bc[kr*nc+j];
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[(i+ic)*N+j+jc] += Cc[i*nc+j];
          }
        }
      }
    }
  }


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
