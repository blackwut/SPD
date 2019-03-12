#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8

void print_matrix(int * m, int rows, int cols)
{
    if (N < 16) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%2d ", m[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {

    int world_rank;
    int world_size;
    int partner_rank;

    int row_id = 3;
    int column_id = 2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size > 2) {
        printf("More then two processes");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    partner_rank = (world_rank + 1) % 2;
    printf("rank: %d\tpartner_rank: %d\n", world_rank, partner_rank);

    MPI_Datatype matrix;
    MPI_Type_contiguous(N * N, MPI_INT, &matrix);
    MPI_Type_commit(&matrix);

    MPI_Datatype row;
    MPI_Type_contiguous(N, MPI_INT, &row);
    MPI_Type_commit(&row);

    MPI_Datatype column;
    MPI_Type_vector(N, 1, N, MPI_INT, &column);
    MPI_Type_commit(&column);

    MPI_Datatype tri_columns;
    MPI_Type_vector(N, 3, N, MPI_INT, &tri_columns);
    MPI_Type_commit(&tri_columns);

    MPI_Datatype up_diagonal;
    MPI_Type_vector(N, 1, N + 1, MPI_INT, &up_diagonal);
    MPI_Type_commit(&up_diagonal);

    MPI_Datatype down_diagonal;
    MPI_Type_vector(N, 1, N - 1, MPI_INT, &down_diagonal);
    MPI_Type_commit(&down_diagonal);


    if (world_rank % 2 == 0) {

        int * m_send = (int *)malloc(N * N * sizeof(int));

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                m_send[i * N + j] = j;
            }
        }

        // print_matrix(m, N, N);

        // Send all matrix values
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
        // Send row `row_id` values
        MPI_Send(m_send + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);
        // Send column `column_id` values
        MPI_Send(m_send + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);
        // Send values of three columns starting from `column_id`
        MPI_Send(m_send + column_id, 1, tri_columns, partner_rank, 0, MPI_COMM_WORLD);
        // Send up_diagonal values
        MPI_Send(m_send, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);
        // Send down_diagonal values
        MPI_Send(m_send + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);

    } else {

        int * m_recv = (int *)malloc(N * N * sizeof(int));
        memset(m_recv, 0, N * N * sizeof(int));

        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received all matrix values\n");
        print_matrix(m_recv, N, N);
        // Check all matrix values
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (m_recv[i * N + j] != j) {
                    printf("Error matrix\n");
                    exit(-1);
                }
            }
        }

        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received row[%d]\n", row_id);
        print_matrix(m_recv, N, N);
        // Check row `row_id` values
        for (int i = 0; i < N; ++i) {
            if (m_recv[i + row_id * N] != i) {
                printf("Error column\n");
                exit(-2);
            }
        }

        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received column[%d]\n", column_id);
        print_matrix(m_recv, N, N);
        // Check column `column_id` values
        for (int i = 0; i < N; ++i) {
            if (m_recv[column_id + i * N] != column_id) {
                printf("Error column\n");
                exit(-3);
            }
        }

        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, tri_columns, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received three columns starting from %d\n", column_id);
        print_matrix(m_recv, N, N);
        // Check values of three columns starting from `column_id`
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (m_recv[column_id + i * N + j] != column_id + j) {
                    printf("Error column\n");
                    exit(-4);
                }
            }
        }

        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received up_diagonal\n");
        print_matrix(m_recv, N, N);
        // Check up_diagonal
        for (int i = 0; i < N; ++i) {
            if (m_recv[i * (N + 1)] != i) {
                printf("Error up_diagonal\n");
                exit(-5);
            }
        }


        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received down_diagonal\n");
        print_matrix(m_recv, N, N);
        // Check down_diagonal
        for (int i = 0; i < N; ++i) {
            if (m_recv[(i + 1) * (N - 1)] != N - i - 1) {
                printf("Error down_diagonal\n");
                exit(-6);
            }
        }
    }

    MPI_Finalize();

    return 0;
}