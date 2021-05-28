//20M19118
//module load intel-mpi
//Compile with mpicxx mpi_openmp_code.cpp -fopenmp -std=c++11
//mpirun -np 4 ./a.out

#include <mpi.h>
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

int main(int argc, char** argv)
{
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    const int N = 256;
    vector<double> A(N * N);
    vector<double> B(N * N);
    vector<double> C(N * N, 0);
    vector<double> subA(N * N / size);
    vector<double> subB(N * N / size);
    vector<double> subC(N * N / size, 0);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[N * i + j] = drand48();
            B[N * i + j] = drand48();
        }
    }

    int offset = N / size * rank;
    for (int i = 0; i < N / size; i++)
        for (int j = 0; j < N; j++)
            subA[N * i + j] = A[N * (i + offset) + j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N / size; j++)
            subB[N / size * i + j] = B[N * i + j + offset];
    int recv_from = (rank + 1) % size;
    int send_to = (rank - 1 + size) % size;

    
    double comp_time = 0, comm_time = 0;
    for (int irank = 0; irank < size; irank++)
    {

        auto tic = chrono::steady_clock::now();
        
        offset = N / size * ((rank + irank) % size);
        int i, j, k;
# pragma omp parallel shared (subA, subB, subC, size, offset) private (i, j, k)
{
# pragma omp for
        for (i = 0; i < N / size; i++)
            for (j = 0; j < N / size; j++)
                for (k = 0; k < N; k++)
                    subC[N * i + j + offset] += subA[N * i + k] * subB[N / size * k + j];
}
        
        auto toc = chrono::steady_clock::now();
        comp_time += chrono::duration<double>(toc - tic).count();
        
        MPI_Send(&subB[0], N * N / size, MPI_DOUBLE, send_to, 0, MPI_COMM_WORLD);
        MPI_Recv(&subB[0], N * N / size, MPI_DOUBLE, recv_from, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        
        tic = chrono::steady_clock::now();
        comm_time += chrono::duration<double>(tic - toc).count();
    }

    MPI_Allgather(&subC[0], N * N / size, MPI_DOUBLE, &C[0], N * N / size, MPI_DOUBLE, MPI_COMM_WORLD);
  
// Record the Error
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[N * i + j] -= A[N * i + k] * B[N * k + j];
    
    double err = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            err += fabs(C[N * i + j]);
    
    if (rank == 0)
    {
        double time = comp_time + comm_time;
        printf("N    : %d\n", N);
        printf("comp : %lf s\n", comp_time);
        printf("comm : %lf s\n", comm_time);
        printf("total: %lf s (%lf GFlops)\n", time, 2. * N * N * N / time / 1e9);
        printf("error: %lf\n", err / N / N);
    }
    MPI_Finalize();
}
