#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEBUG 0

void fill_matrix_diagonal(int * m, size_t dim, int val)
{
    for (size_t i = 0; i < dim; ++i) {
        m[i * dim + i] = val;
    }
}

void fill_matrix(int * m, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            m[i * dim + j] = i * dim + j + 1;
        }
    }
}

void print_matrix(int * m, size_t rows, size_t cols, const char * message)
{
    printf("%s\n", message);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            printf("%4d ", m[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int is_square(int x)
{
    int y = 1;
    while (y * y <= x) {
        if (y * y == x) {
            return 1;
        }
        ++y;
    }
    return 0;
}

int main(int argc, char** argv)
{
    int world_rank;
    int world_size;

    size_t N = 0;

    int * A = NULL;
    int * B = NULL;
    int * C = NULL;

    int * block_A = NULL; // rows
    int * block_B = NULL; // columns
    int * block_C = NULL; // block

    int * rows_counts     = NULL;
    int * rows_displays   = NULL;
    int * cols_counts     = NULL;
    int * cols_displays   = NULL;
    int * blocks_counts   = NULL;
    int * blocks_displays = NULL;

    if (argc < 2) {
        exit(-1);
    }

    N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (!is_square(world_size) || N % world_size > 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t blocks_per_dim = (size_t)sqrt(world_size);
    size_t block_dim = N / blocks_per_dim;

    MPI_Datatype matrix_block;
    MPI_Type_vector(block_dim, block_dim, N, MPI_INT, &matrix_block);
    MPI_Type_create_resized(matrix_block, 0, sizeof(int), &matrix_block);
    MPI_Type_commit(&matrix_block);

    MPI_Datatype block;
    MPI_Type_contiguous(block_dim * block_dim, MPI_INT, &block);
    MPI_Type_commit(&block);

    MPI_Datatype partition;
    MPI_Type_contiguous(block_dim * N, MPI_INT, &partition);
    MPI_Type_commit(&partition);

    MPI_Datatype columns;
    MPI_Type_vector(N, block_dim, N, MPI_INT, &columns);
    MPI_Type_create_resized(columns, 0, sizeof(int), &columns);
    MPI_Type_commit(&columns);

    MPI_Datatype rows;
    MPI_Type_contiguous(block_dim * N, MPI_INT, &rows);
    MPI_Type_create_resized(rows, 0, sizeof(int), &rows);
    MPI_Type_commit(&rows);

    if (world_rank == 0) {
        A = (int *)calloc(N * N, sizeof(int));
        B = (int *)calloc(N * N, sizeof(int));
        C = (int *)calloc(N * N, sizeof(int));

        fill_matrix(A, N);
        fill_matrix_diagonal(B, N, 4);

        // counts and displays
        rows_counts     = (int *)calloc(world_size, sizeof(int));
        rows_displays   = (int *)calloc(world_size, sizeof(int));

        cols_counts     = (int *)calloc(world_size, sizeof(int));
        cols_displays   = (int *)calloc(world_size, sizeof(int));

        blocks_counts   = (int *)calloc(world_size, sizeof(int));
        blocks_displays = (int *)calloc(world_size, sizeof(int));

        for (int i = 0; i < world_size; ++i) {
            rows_counts[i] = 1;
            rows_displays[i] = (i / blocks_per_dim) * block_dim * N;

            cols_counts[i] = 1;
            cols_displays[i] = (i / blocks_per_dim) * block_dim;
        }

        for (int i = 0; i < blocks_per_dim; ++i) {
            for (int j = 0; j < blocks_per_dim; ++j) {
                blocks_counts[i * blocks_per_dim + j] = 1;
                blocks_displays[i * blocks_per_dim + j] = j * block_dim + i * block_dim * N;
            }
        }
    }

#if DEBUG
    if (world_rank == 0) {

        printf("block_dim: %zu\nblocks_per_dim: %zu\n", block_dim, blocks_per_dim);

        print_matrix(A, N, N, "Matrix A");
        print_matrix(B, N, N, "Matrix B");

        // print_matrix(rows_counts, 1, world_size, "rows_counts");
        print_matrix(rows_displays, 1, world_size, "rows_displays");
        // print_matrix(cols_counts, 1, world_size, "cols_counts");
        print_matrix(cols_displays, 1, world_size, "cols_displays");
        // print_matrix(blocks_counts, 1, world_size, "blocks_counts");
        print_matrix(blocks_displays, blocks_per_dim, blocks_per_dim, "blocks_displays");
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    block_A = (int *)calloc(block_dim * N, sizeof(int));
    block_B = (int *)calloc(N * block_dim, sizeof(int));
    block_C = (int *)calloc(block_dim * block_dim, sizeof(int));

    MPI_Scatterv(A, rows_counts, rows_displays, rows,
                 block_A, 1, partition,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, cols_counts, cols_displays, columns,
                 block_B, 1, partition,
                 0, MPI_COMM_WORLD);

    for (size_t i = 0; i < block_dim; ++i) {
        for (size_t j = 0; j < block_dim; ++j) {
            int c = 0;
            for (size_t k = 0; k < N; ++k) {
                c += block_A[i * N + k] * block_B[k * block_dim + j];
            }
            block_C[i * block_dim + j] = c;
        }
    }

    MPI_Gatherv(block_C, 1, block,
                C, blocks_counts, blocks_displays, matrix_block,
                0, MPI_COMM_WORLD);

#if DEBUG
    for (size_t i = 0; i < world_size; ++i) {
        if (world_rank == i) {
            printf("\n*** RANK %zu ***\n", i);
            print_matrix(block_A, block_dim, N, "block_A");
            print_matrix(block_B, N, block_dim, "block_B");
            print_matrix(block_C, block_dim, block_dim, "block_C");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    if (world_rank == 0) {
        print_matrix(C, N, N, "C");
    }

    MPI_Finalize();

    return 0;
}