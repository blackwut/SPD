#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 128

void fill_matrix_diagonal(int * m, size_t dim, int val)
{
    for (size_t i = 0; i < dim; ++i) {
        m[i * dim + i] = val;
    }
}

void fill_matrix(int * m, size_t dim)
{
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            m[i * dim + j] = j;
        }
    }
}

int main(int argc, char** argv)
{
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size > 4) {
        printf("More then 4 processes");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("rank: %d\n", world_rank);

    MPI_Datatype matrix;
    MPI_Type_contiguous(N * N, MPI_INT, &matrix);
    MPI_Type_commit(&matrix);

    // Business code of matrix multiplication

    MPI_Finalize();

    return 0;
}