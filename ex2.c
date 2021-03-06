#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PRINTABLE_DIM      (16)
#define N                       (8)
//#define N                 (1 << 14) // 16384 x 16384 = 268435456 ~ 2GB in total
#define SEND_RECEIVE            (1)


void fill_matrix(int * m, size_t dim)
{
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            m[i * dim + j] = j;
        }
    }
}


int compute_value(int val)
{
    return val * 2 + 1;
}


void update_matrix(int * m, int * m_send,
                   size_t n, size_t blocklength, size_t stride)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < blocklength; ++j) {
            const size_t index = i * stride + j;
            m_send[index] = compute_value(m[index]);
        }
    }
}


void check_matrix(int * m, int * m_recv,
                  size_t n, size_t blocklength, size_t stride,
                  const char * message)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < blocklength; ++j) {
            const size_t index = i * stride + j;
            if (compute_value(m[index]) != m_recv[index]) {
                fprintf(stderr, "Error: %s\n", message);
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }
}


void check_reverse(int * m, int * m_recv, size_t n,
                  const char * message)
{
    const size_t dim = n * n;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const size_t index = i * n + j;
            if (m[index] != m_recv[dim - index - 1]) {
                fprintf(stderr, "Error: %s\n", message);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
        }
    }
}


void check_transpose(int * m, int * m_recv, size_t n,
                     const char * message)
{
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (m[i * n + j] != m_recv[j * n + i]) {
                fprintf(stderr, "Error: %s\n", message);
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
        }
    }
}


void print_matrix(int * m, size_t dim)
{
    if (dim < MAX_PRINTABLE_DIM) {
        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                printf("%3d ", m[i * dim + j]);
            }
            printf("\n");
        }
    }
}


int main(int argc, char * argv[])
{
    int world_rank;
    int world_size;
    size_t partner_rank;

    int * m_send;
    int * m_recv;
    size_t row_id;
    size_t column_id;
    size_t columns = 3;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size > 2) {
        printf("More then two processes");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    partner_rank = (world_rank + 1) % 2;
    printf("rank: %d\tpartner_rank: %zu\n", world_rank, partner_rank);

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
    MPI_Type_vector(N, columns, N, MPI_INT, &three_columns);
    MPI_Type_commit(&three_columns);

    MPI_Datatype up_diagonal;
    MPI_Type_vector(N, 1, N + 1, MPI_INT, &up_diagonal);
    MPI_Type_commit(&up_diagonal);

    MPI_Datatype down_diagonal;
    MPI_Type_vector(N, 1, N - 1, MPI_INT, &down_diagonal);
    MPI_Type_commit(&down_diagonal);

    MPI_Datatype reverse;
    MPI_Type_vector(N * N, 1, -1, MPI_INT, &reverse);
    MPI_Type_commit(&reverse);

    MPI_Datatype transpose_column;
    MPI_Type_create_resized(column, 0, sizeof(int), &transpose_column);
    MPI_Type_commit(&transpose_column);

    m_send = (int *)calloc(N * N, sizeof(int));
    m_recv = (int *)calloc(N * N, sizeof(int));
    row_id = 2;
    column_id = 3;

    if (world_rank % 2 == 0) {

       fill_matrix(m_send, N);

        // matrix
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send, 1, matrix, partner_rank, 0,
                     m_recv, 1, matrix, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send, m_recv, N, N, N, "matrix");
        printf("PingPong Matrix completed!\n\n");

        // row
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send + row_id * N, 1, row, partner_rank, 0,
                     m_recv + row_id * N, 1, row, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
        MPI_Send(m_send + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send + row_id * N, m_recv + row_id * N, 1, N, 1, "row");
        printf("PingPong row completed!\n\n");

        // column
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send + column_id, 1, column, partner_rank, 0,
                     m_recv + column_id, 1, column, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send + column_id, m_recv + column_id, N, 1, N, "column");
        printf("PingPong column completed!\n\n");

        // three columns
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send + column_id, 1, three_columns, partner_rank, 0,
                     m_recv + column_id, 1, three_columns, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 3, three_columns, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send + column_id, m_recv + column_id, N, columns, N, "three_columns");
        printf("PingPong three_columns completed!\n\n");

        // up_diagonal
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send, 1, up_diagonal, partner_rank, 0,
                     m_recv, 1, up_diagonal, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send, m_recv, N, 1, N + 1, "up_diagonal");
        printf("PingPong up_diagonal completed!\n\n");

        // down_diagonal
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send + N - 1, 1, down_diagonal, partner_rank, 0,
                     m_recv + N - 1, 1, down_diagonal, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_matrix(m_send + N - 1, m_recv + N - 1, N, 1, N - 1, "down_diagonal");
        printf("PingPong down_diagonal completed!\n\n");

        // reverse
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send, 1, matrix, partner_rank, 0,
                     m_recv, 1, matrix, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_reverse(m_send, m_recv, N, "reverse");
        printf("PingPong reverse completed!\n\n");

        // transpose
#if SEND_RECEIVE
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Sendrecv(m_send, 1, matrix, partner_rank, 0,
                     m_recv, 1, matrix, partner_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

#else
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#endif
        print_matrix(m_recv, N);
        check_transpose(m_send, m_recv, N, "transpose");
        printf("PingPong transpose completed!\n\n");

    } else {

        // matrix
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv, m_send, N, N, N);
#if SEND_RECEIVE
        MPI_Rsend(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // row
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv + row_id * N, m_send + row_id * N, 1, N, 1);
#if SEND_RECEIVE
        MPI_Rsend(m_send + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send + row_id * N, 1, row, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // column
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv + column_id, m_send + column_id, 1, 1, N);
#if SEND_RECEIVE
        MPI_Rsend(m_send + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send + column_id, 1, column, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // three column
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv + column_id, m_send + column_id, 1, columns, N);
#if SEND_RECEIVE
        MPI_Rsend(m_send + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send + column_id, 1, three_columns, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // up_diagonal
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv, m_send, N, 1, N + 1);
#if SEND_RECEIVE
        MPI_Rsend(m_send, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send, 1, up_diagonal, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // down_diagonal
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        update_matrix(m_recv, m_send, N, 1, N - 1);
#if SEND_RECEIVE
        MPI_Rsend(m_send + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_send + N - 1, 1, down_diagonal, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // reverse
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv + N * N - 1, 1, reverse, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if SEND_RECEIVE
        MPI_Rsend(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD);
#endif

        // transpose
        memset(m_recv, 0, N * N * sizeof(int));
        MPI_Recv(m_recv, 1, matrix, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#if SEND_RECEIVE
        MPI_Rsend(m_recv, N, transpose_column, partner_rank, 0, MPI_COMM_WORLD);
#else
        MPI_Send(m_recv, N, transpose_column, partner_rank, 0, MPI_COMM_WORLD);
#endif
    }

    MPI_Type_free(&matrix);
    MPI_Type_free(&row);
    MPI_Type_free(&column);
    MPI_Type_free(&three_columns);
    MPI_Type_free(&up_diagonal);
    MPI_Type_free(&down_diagonal);
    MPI_Type_free(&reverse);
    MPI_Type_free(&transpose_column);

    MPI_Finalize();

    return 0;
}