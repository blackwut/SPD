#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 8

void fill_matrix(int * m, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            m[i * cols + j] = j;
        }
    }
}

int compute_value(int val)
{
    return val * 2 + 1;
}

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

int main(int argc, char** argv)
{
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

    MPI_Datatype three_columns;
    MPI_Type_vector(N, 3, N, MPI_INT, &three_columns);
    MPI_Type_commit(&three_columns);

    MPI_Datatype up_diagonal;
    MPI_Type_vector(N, 1, N + 1, MPI_INT, &up_diagonal);
    MPI_Type_commit(&up_diagonal);

    MPI_Datatype down_diagonal;
    MPI_Type_vector(N, 1, N - 1, MPI_INT, &down_diagonal);
    MPI_Type_commit(&down_diagonal);


    if (world_rank % 2 == 0) {

        int * m_send = (int *)malloc(N * N * sizeof(int));
        int * m_recv = (int *)malloc(N * N * sizeof(int));
        fill_matrix(m_send, N, N);

        // matrix
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int index = i * N + j;
                if (m_recv[index] != compute_value(m_send[index])) {
                    printf("Error matrix\n");
                    exit(-1);
                }
            }
        }

        // row
        MPI_Send(m_send + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            int index = i + row_id * N;
            if (m_recv[index] != compute_value(m_send[index])) {
                printf("Error row\n");
                exit(-2);
            }
        }

        // column
        MPI_Send(m_send + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            int index = i * N + column_id;
            if (m_recv[index] != compute_value(m_send[index])) {
                printf("Error column\n");
                exit(-3);
            }
        }

        // three columns
        MPI_Send(m_send + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 3; ++j) {
                int index = i * N + column_id + j;
                if (m_recv[index] != compute_value(m_send[index])) {
                    printf("Error three column\n");
                    exit(-4);
                }
            }
        }

        // up_diagonal
        MPI_Send(m_send, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            int index = i * (N + 1);
            if (m_recv[index] != compute_value(m_send[index])) {
                printf("Error up_diagonal\n");
                exit(-5);
            }
        }

        // down_diagonal
        MPI_Send(m_send + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print_matrix(m_recv, N, N);
        // Check
        for (int i = 0; i < N; ++i) {
            int index = (i + 1) * (N - 1);
            if (m_recv[index] != compute_value(m_send[index])) {
                printf("Error down_diagonal\n");
                exit(-5);
            }
        }

    } else {

        int * m_recv = (int *)malloc(N * N * sizeof(int));

        // matrix
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int index = i * N + j;
                m_recv[index] = compute_value(m_recv[index]);
            }
        }
        MPI_Send(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);

        // row
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            int index = i + row_id * N;
            m_recv[index] = compute_value(m_recv[index]);
        }
        MPI_Send(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);

        // column
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            int index = i * N + column_id;
            m_recv[index] = compute_value(m_recv[index]);
        }
        MPI_Send(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);

        // three column
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < 3; ++j) {
                int index = i * N + column_id + j;
                m_recv[index] = compute_value(m_recv[index]);
            }
        }
        MPI_Send(m_recv + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD);

        // up_diagonal
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            int index = i * (N + 1);
            m_recv[index] = compute_value(m_recv[index]);
        }
        MPI_Send(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);

        // down_diagonal
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < N; ++i) {
            int index = (i + 1) * (N - 1);
            m_recv[index] = compute_value(m_recv[index]);
        }
        MPI_Send(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}