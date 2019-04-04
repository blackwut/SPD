#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>

#define DEBUG 0

#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    int d;
    double * data;
    double distance;
    int centroid;
} DataRow;


double random_num(int n)
{
#if 0
    return arc4random_uniform(n);
#else
    return random() / pow(2, 31) * n;
#endif
}


int data_row_size(size_t d)
{
    int size = 0;
    size += sizeof(int);
    size += sizeof(double) * d;
    size += sizeof(double);
    size += sizeof(int);
    return size;
}

DataRow * data_row_alloc(size_t d, size_t n)
{
    DataRow * dataset = (DataRow *)calloc(n, sizeof(DataRow));
    for (size_t i = 0; i < n; ++i) {
        dataset[i].d = d;
        dataset[i].data = (double *)calloc(d, sizeof(double));
        dataset[i].distance = DBL_MAX;
        dataset[i].centroid = -1;
    }
    return dataset;
}

void data_row_free(DataRow * dataset, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        free(dataset[i].data);
    }
    free(dataset);
}

char * data_row_alloc_raw(size_t d, size_t n)
{
    return calloc(data_row_size(d) * n, sizeof(char));
}

void data_row_free_raw(char * raw)
{
    free(raw);
}

void pack_data_rows(char * raw, const DataRow * rows, int size, size_t n, MPI_Comm comm)
{
    int pos = 0;

    for (size_t i = 0; i < n; ++i) {
        MPI_Pack(&rows[i].d, 1, MPI_INT, raw, size, &pos, comm);
        MPI_Pack(rows[i].data, rows[i].d, MPI_DOUBLE, raw, size, &pos, comm);
        MPI_Pack(&rows[i].distance, 1, MPI_DOUBLE, raw, size, &pos, comm);
        MPI_Pack(&rows[i].centroid, 1, MPI_INT, raw, size, &pos, comm);
    }
}

void unpack_data_rows(DataRow * dataset, const char * raw, int size, size_t d, size_t n, MPI_Comm comm)
{
    int pos = 0;

    for (size_t i = 0; i < n; ++i) {
        MPI_Unpack(raw, size, &pos, &dataset[i].d, 1, MPI_INT, comm);
        MPI_Unpack(raw, size, &pos, dataset[i].data, d, MPI_DOUBLE, comm);
        MPI_Unpack(raw, size, &pos, &dataset[i].distance, 1, MPI_DOUBLE, comm);
        MPI_Unpack(raw, size, &pos, &dataset[i].centroid, 1, MPI_INT, comm);
    }
}

void read_data(FILE * f, double * data, size_t d)
{
    for (size_t i = 0; i < d; ++i) {
        fscanf(f, "%le", &data[i]);
    }
}

void read_dataset(FILE * f, DataRow * dataset, size_t d, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        read_data(f, dataset[i].data, d);
    }
}

void write_data(const double * data, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        printf("%10e\t", data[i]); 
    }
    printf("\n");
    fflush(stdout);
}

void write_centroids(const char * message, const DataRow * centroids, size_t d, size_t k, size_t iteration)
{
    printf("# %s %zu\n", message, iteration);
    printf("# centroids\n");
    for (size_t i = 0; i < k; ++i) {
        write_data(centroids[i].data, d);
    }
}

void select_random_points(DataRow * centroids, size_t d, size_t n, size_t k)
{
    for (size_t i = 0; i < k; ++i) {
        centroids[i].d = d;
        for (size_t j = 0; j < d; ++j){
            centroids[i].data[j] = random_num(1);
        }
        centroids[i].distance = 0;
        centroids[i].centroid = -1;
    }
}

void select_random_centroids(DataRow * centroids, const DataRow * dataset, size_t n, size_t k)
{
    size_t * random_list = (size_t *)calloc(k, sizeof(size_t));
    for (size_t i = 0; i < k; ++i) {
        int ok = 0;
        while (!ok) {
            ok = 1;
            const size_t random = random_num(n);
            for (size_t j = 0; j < i; ++j) {
                if (random == random_list[j]) {
                    ok = 0;
                    break;
                }
            }
            random_list[i] = random;
        }
    }

    for (size_t i = 0; i < k; ++i) {
        const size_t random = random_list[i];
        centroids[i].d = dataset[random].d;
        for (size_t j = 0; j < dataset[random].d; ++j){
            centroids[i].data[j] = dataset[random].data[j];
        }
        centroids[i].distance = 0;
        centroids[i].centroid = -1;
    }

    free(random_list);
}

void clear_centroids_data(double * data, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0.0;
    }
}

void clear_centroids_hits(int * hits, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        hits[i] = 0;
    }
}

void sum_data(double * a, double * b, size_t d)
{
    for (size_t i = 0; i < d; ++i) {
        a[i] += b[i];
    }
}

double euclidean_distance(double * a, double * b, size_t d)
{
    double sum = 0.0;

    for (size_t i = 0; i < d; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

void compute_minimum_distance(DataRow * row, const DataRow * centroids, size_t d, size_t k, MPI_Comm comm)
{
    double min = DBL_MAX;
    int closest = -1;

    for (size_t i = 0; i < k; ++i) {
        const double distance = euclidean_distance(row->data, centroids[i].data, d);
        if (distance < min) {
            min = distance;
            closest = i;
        }
    }

    if (closest == -1) {
        MPI_Abort(comm, 127);
    }

    row->distance = min;
    row->centroid = closest;
}

void update_centroids(DataRow * centroids, const double * data, const int * hits, size_t d, size_t k)
{
    for (size_t i = 0; i < k; ++i) {
        if (hits[i] > 0) {
            for (size_t j = 0; j < d; ++j) {
                centroids[i].data[j] = data[i * d + j] / hits[i];
            }
        }
    }
}

void print_debug(const char * message, MPI_Comm comm)
{
#if DEBUG
    int rank;
    int size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("%s\n", message);
    }
    fflush(stdout);
    MPI_Barrier(comm);

    for (size_t i = 0; i < size; ++i) {
        if (rank == i) {
            printf("rank: %d\n", rank);
        }
        fflush(stdout);
        MPI_Barrier(comm);
    }
    fflush(stdout);
#endif
}

void print_debug_centroids(const char * message, DataRow * centroids, size_t d, size_t k, MPI_Comm comm)
{
#if DEBUG
    int rank;
    int size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("%s\n", message);
    }
    fflush(stdout);
    MPI_Barrier(comm);

    for (size_t i = 0; i < size; ++i) {
        if (rank == i) {
            printf("rank: %d\n", rank);
            for (size_t r = 0; r < k; ++r) {
                for (size_t c = 0; c < d; ++c) {
                    printf("%10e\t", centroids[r].data[c]);
                }
                printf("\n");
            }
        }
        fflush(stdout);
        MPI_Barrier(comm);
    }
    fflush(stdout);
#endif
}

void print_debug_centroids_data(const char * message, double * centroids_data, int * centroids_hits, size_t d, size_t k, MPI_Comm comm)
{
#if DEBUG
    int rank;
    int size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("%s\n", message);
    }
    fflush(stdout);
    MPI_Barrier(comm);

    for (size_t i = 0; i < size; ++i) {
        if (rank == i) {
            printf("rank: %d\n", rank);
            for (size_t r = 0; r < k; ++r) {
                printf("count: %d\t", centroids_hits[r]);
                for (size_t c = 0; c < d; ++c) {
                    printf("%10e\t", centroids_data[r * d + c]);
                }
                printf("\n");
            }
        }
        fflush(stdout);
        MPI_Barrier(comm);
    }
    fflush(stdout);
#endif
}

void kmeans(char * filename, size_t d, size_t n, size_t k, size_t iterations, MPI_Comm comm)
{
    int rank;
    int size;
    MPI_Comm c;

    size_t it_count = 0;
    DataRow * dataset = NULL;
    char * dataset_raw = NULL;
    size_t dataset_count = 0;
    size_t dataset_size = 0;
    size_t dataset_count_local = 0;
    DataRow * centroids = NULL;
    char * centroids_raw = NULL;
    size_t centroids_count = 0;
    size_t centroids_size = 0;
    double * centroids_data = NULL;
    int centroids_data_count = 0;
    int * centroids_hits = NULL;
    int centroids_hits_count = 0;

    MPI_Comm_dup(comm, &c);
    MPI_Comm_rank(c, &rank);
    MPI_Comm_size(c, &size);

    // Data structures of MPI_Scatterv
    const int dataRowSize = data_row_size(d);
    int * counts = (int *)calloc(size, sizeof(int));
    int * displays = (int *)calloc(size, sizeof(int));

    for (size_t i = 0; i < size; ++i) {
        int elems = (n + size - 1) / size;
        counts[i] = (elems * (i + 1) < n) ? elems : n - (elems * i);
        counts[i] *= dataRowSize;
    }

    int displaysAcc = 0;
    displays[0] = 0;
    for (size_t i = 1; i < size; ++i) {
        displaysAcc += counts[i - 1];
        displays[i] = displaysAcc;
    }

    dataset_count = (rank == 0 ? n : counts[rank] / dataRowSize);
    dataset_size = dataset_count * dataRowSize;
    dataset_count_local = counts[rank] / dataRowSize;
    centroids_count = k;
    centroids_size = centroids_count * dataRowSize;
    centroids_data_count = d * k;
    centroids_hits_count = k;

    dataset = data_row_alloc(d, dataset_count);
    dataset_raw = data_row_alloc_raw(d, dataset_count);
    centroids = data_row_alloc(d, centroids_count);
    centroids_raw = data_row_alloc_raw(d, centroids_size);
    centroids_data = (double *)calloc(centroids_data_count, sizeof(double));
    centroids_hits = (int *)calloc(centroids_hits_count, sizeof(int));

    if (rank == 0) {
        FILE * f = fopen(filename, "r");
        read_dataset(f, dataset, d, n);
        fclose(f);
        pack_data_rows(dataset_raw, dataset, dataset_size, dataset_count, c);

        select_random_points(centroids, d, n, k);
        //select_random_centroids(centroids, dataset, n, k);
        pack_data_rows(centroids_raw, centroids, centroids_size, centroids_count, c);
    }
    

    MPI_Scatterv(dataset_raw, counts, displays, MPI_CHAR, dataset_raw, dataset_size, MPI_CHAR, 0, c);
    MPI_Bcast(centroids_raw, centroids_size, MPI_CHAR, 0, c);

    unpack_data_rows(dataset, dataset_raw, dataset_size, d, dataset_count_local, c);
    unpack_data_rows(centroids, centroids_raw, centroids_size, d, centroids_count, c);
    print_debug_centroids("centroids", centroids, d, k, c);


    double MSE_new = DBL_MAX;
    double MSE_old;
    double MSE_local;

    do {
        MSE_old = MSE_new;
        MSE_local = 0.0;

        if (rank == 0) {
            write_centroids("Iteration", centroids, d, k, it_count);
        }

        clear_centroids_data(centroids_data, centroids_data_count);
        clear_centroids_hits(centroids_hits, centroids_hits_count);
        print_debug_centroids_data("CLEAR", centroids_data, centroids_hits, d, k, c);

        for (size_t i = 0; i < dataset_count_local; ++i) {
            DataRow * point = &dataset[i];

            compute_minimum_distance(point, centroids, d, k, c);
            const int closest = point->centroid;
            sum_data(centroids_data + (closest * d), point->data, d);
            centroids_hits[closest] += 1;
            MSE_local += point->distance * point->distance;
        }

        MPI_Allreduce(MPI_IN_PLACE, centroids_data, centroids_data_count, MPI_DOUBLE, MPI_SUM, c);
        MPI_Allreduce(MPI_IN_PLACE, centroids_hits, centroids_hits_count, MPI_DOUBLE, MPI_SUM, c);
        MPI_Allreduce(&MSE_local, &MSE_new, 1, MPI_DOUBLE, MPI_SUM, c);

        print_debug_centroids_data("REDUCE", centroids_data, centroids_hits, d, k, c);
        update_centroids(centroids, centroids_data, centroids_hits, d, k);
        print_debug_centroids_data("UPDATE", centroids_data, centroids_hits, d, k, c);


        it_count++;
    } while (MSE_new < MSE_old && it_count < iterations);

    if (rank == 0) {
        write_centroids("Iteration", centroids, d, k, it_count);
        write_centroids("Final result", centroids, d, k, it_count);
    }

    // MPI_Barrier(c);

    // printf("rank %d\n", rank);

    // MPI_Barrier(c);

    // MPI_Comm_free(&c);
    if (counts) free(counts);
    if (displays) free(displays);
    if (dataset) data_row_free(dataset, n); // free EXC_BAD_ACCESS
    if (dataset_raw) free(dataset_raw);
    if (centroids) data_row_free(centroids, k);
    if (centroids_raw) free(centroids_raw);
    if (centroids_data) free(centroids_data);
    if (centroids_hits) free(centroids_hits);
}

int main(int argc, char * argv[])
{
    if (argc < 6) {
        fprintf(stderr, "./kmeans [file.dat] [d] [n] [k] [iterations]\n");
        exit(1);
    }

    int d = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);
    int iterations = atoi(argv[5]);

    srandom(100);

    MPI_Init(&argc, &argv);

    kmeans(argv[1], d, n, k, iterations, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
