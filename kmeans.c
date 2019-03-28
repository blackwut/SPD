#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

typedef struct {
    int d;
    double * data;
    double distance;
    int centroid;
} DataRow;


int data_row_size(size_t d)
{
    int size = 0;
    size += sizeof(int);
    size += sizeof(double) * d;
    size += sizeof(double);
    size += sizeof(int);
    return size;
}

char * pack_data_rows(DataRow * row, size_t n, MPI_Comm comm)
{
    int pos = 0;
    int size = 0;

    size = data_row_size(row[0].d) * n; // Assuming rows with same "d" value
    char * data = (char *)calloc(size, sizeof(char));

    for (size_t i = 0; i < n; ++i) {
        MPI_Pack(&row[i].d, 1, MPI_INT, data, size, &pos, comm);
        MPI_Pack(row[i].data, row[i].d, MPI_DOUBLE, data, size, &pos, comm);
        MPI_Pack(&row[i].distance, 1, MPI_DOUBLE, data, size, &pos, comm);
        MPI_Pack(&row[i].centroid, 1, MPI_INT, data, size, &pos, comm);
    }

    return data;
}

DataRow * unpack_data_rows(char * data, int size, size_t n, MPI_Comm comm)
{
    int pos = 0;
    DataRow * row = (DataRow *)calloc(n, sizeof(DataRow));

    for (size_t i = 0; i < n; ++i) {
        MPI_Unpack(data, size, &pos, &row[i].d, 1, MPI_INT, comm);
        row[i].data = (double *)calloc(row[i].d, sizeof(double));
        MPI_Unpack(data, size, &pos, row[i].data, row[i].d, MPI_DOUBLE, comm);
        MPI_Unpack(data, size, &pos, &row[i].distance, 1, MPI_DOUBLE, comm);
        MPI_Unpack(data, size, &pos, &row[i].centroid, 1, MPI_INT, comm);
    }

    return row;
}

void read_data(double * data, size_t d)
{
    for (size_t i = 0; i < d; ++i) {
        scanf("%le", &data[i]);
    }
}

DataRow * read_dataset(size_t d, size_t n)
{
    DataRow * dataset = (DataRow *)calloc(n, sizeof(DataRow));
    for (size_t i = 0; i < n; ++i) {
        dataset[i].d = d;
        dataset[i].data = (double *)calloc(d, sizeof(double));
        dataset[i].distance = DBL_MAX;
        dataset[i].centroid = -1;
    }

    for (size_t i = 0; i < n; ++i) {
        read_data(dataset[i].data, d);
    }

    return dataset;
}

void write_data(double * data, size_t d)
{
    for (size_t i = 0; i < d; ++i) {
        printf("%10e\t", data[i]); 
    }
    printf("\n");
}

void write_centroids(double * data, size_t d, size_t n, size_t iteration)
{
    printf("# Iteration %zu\n", iteration);
    printf("# centroids\n");
    for (size_t i = 0; i < n; ++i) {
        write_data(data + i * d, d);
    }
}


void kmeans(size_t d, size_t n, size_t k, MPI_Comm comm)
{
    int rank;
    int size;
    MPI_Comm c;

    DataRow * dataset = NULL;   // used by the root only
    char * dataset_raw = NULL;  // used by the root only
    DataRow * rows = NULL;
    char * raw = NULL;

    MPI_Comm_dup(comm, &c);
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &size);

    const int dataRowSize = data_row_size(d);
    int * counts = (int *)calloc(size, sizeof(int));
    int * displays = (int *)calloc(size, sizeof(int));

    for (size_t i = 0; i < size; ++i) {
        int c = (n + size - 1) / size;
        counts[i] = (c * (i + 1) < n) ? c : n - (c * i);
        counts[i] *= dataRowSize;
    }

    int displaysAcc = 0;
    displays[0] = 0;
    for (size_t i = 1; i < size; ++i) {
        displaysAcc += counts[i - 1];
        displays[i] = displaysAcc;
    }

    if (rank == 0) {
        for (size_t i = 0; i < size; ++i) 
            printf("%d\n", counts[i]);
    }
    MPI_Barrier(c);

    if (rank == 0) {
        dataset = read_dataset(d, n);
        dataset_raw = pack_data_rows(dataset, n, c);
    }
    
    raw = (char *)calloc(dataRowSize * counts[rank], sizeof(char));
    MPI_Scatterv(dataset_raw, counts, displays, MPI_CHAR, raw, counts[rank], MPI_CHAR, 0, c);

    rows = unpack_data_rows(raw, counts[rank], counts[rank] / dataRowSize, comm);

    for (size_t i = 0; i < size; ++i) {
        if (i == rank) {
            for (size_t j = 0; j < counts[rank] / dataRowSize; ++j) {
                write_data(rows[j].data, rows[j].d);
            }
        }
        MPI_Barrier(c);
    }
}

int main(int argc, char** argv)
{
    int d = 2;
    int n = 4;
    int k = 3;

    MPI_Init(&argc, &argv);
    kmeans(d, n, k, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
