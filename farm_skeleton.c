#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <unistd.h>

#define STREAM_ELEM_SIZE 10
#define MAX_STREAM_COUNT 2

#define STREAM_ELEM_EOS 3

// #define FARM_MODE_RR        1
// #define FARM_MODE_EXPLICIT  2


typedef int (*streamGenerateFun)(void *);
typedef void (*streamComputeFun)(void *);
typedef void (*streamFinalizeFun)(void *);
typedef void * (*streamElemAllocFun)();
typedef void (*streamElemFreeFun)(void *);


void * stream_in_alloc()
{
    return calloc(STREAM_ELEM_SIZE, sizeof(int));
}

void stream_in_free(void * stream_elem)
{
    free(stream_elem);
}

int stream_generate(void * stream_elem)
{
    static size_t count = 0;

    int * elem = (int *)stream_elem;
    if (count++ < MAX_STREAM_COUNT) {
        for (size_t i = 0; i < STREAM_ELEM_SIZE; ++i) {
            elem[i] = STREAM_ELEM_SIZE;
        }
        return !STREAM_ELEM_EOS;
    }

    return STREAM_ELEM_EOS;
}

void stream_compute(void * stream_elem)
{
    int * elem = (int *)stream_elem;
    for (size_t i = 0; i < STREAM_ELEM_SIZE; ++i) {
        elem[i] += 2;
    }
}

void stream_sleep_compute(void * stream_elem)
{
    static int seeded = 0;
    if (seeded == 0) {
        srand(time(NULL));
        seeded = 1;
    }

    stream_compute(stream_elem);

    sleep(1);
}

void stream_random_sleep_compute(void * stream_elem)
{
    static int seeded = 0;
    if (seeded == 0) {
        srand((unsigned int)stream_elem);
        seeded = 1;
    }

    stream_compute(stream_elem);

    int sleep_time = rand() / (float)RAND_MAX * 2000000;
    usleep(sleep_time);
}

void stream_finalize(void * stream_elem)
{
    static size_t count = 0;

    int * elem = (int *)stream_elem;
    printf("%zu) ", count++);
    for (size_t i = 0; i < STREAM_ELEM_SIZE; ++i) {
        printf("%d ", elem[i]);
    }
    printf("\n");

    fflush(stdout);
}

void emitter_handler(streamGenerateFun stream_generate,
                     MPI_Datatype elem_datatype,
                     streamElemAllocFun elem_alloc,
                     streamElemFreeFun elem_free,
                     int nWorkers,
                     MPI_Comm comm_e2w)
{
    void * elem = elem_alloc();
    int i = 0;

    while (stream_generate(elem) != STREAM_ELEM_EOS) { 
        MPI_Ssend(elem, 1, elem_datatype, i, 0, comm_e2w);
        i = (i + 1) % nWorkers;
    }

    for (i = 0; i < nWorkers; ++i) {
        MPI_Ssend(NULL, 0, elem_datatype, i, 0, comm_e2w);
    }

    elem_free(elem);
}

void emitter_handler_issend(streamGenerateFun stream_generate,
                            MPI_Datatype elem_datatype,
                            streamElemAllocFun elem_alloc,
                            streamElemFreeFun elem_free,
                            int nWorkers,
                            MPI_Comm comm_e2w)
{
    int end_of_stream = !STREAM_ELEM_EOS;
    int request_index = 0;
    MPI_Request * requests = (MPI_Request *)calloc(nWorkers, sizeof(MPI_Request));
    void * elem = elem_alloc();

    for (int i = 0; i < nWorkers; ++i) {
        requests[i] = MPI_REQUEST_NULL;
    }

    for (int i = 0; i < nWorkers; ++i) {
        if ( (end_of_stream = stream_generate(elem)) != STREAM_ELEM_EOS) {
            MPI_Issend(elem, 1, elem_datatype, i, 0, comm_e2w, &requests[i]);
        }
    }

    if (end_of_stream != STREAM_ELEM_EOS) {
        while (stream_generate(elem) != STREAM_ELEM_EOS) {
            MPI_Waitany(nWorkers, requests, &request_index, MPI_STATUSES_IGNORE);
            MPI_Issend(elem, 1, elem_datatype, request_index, 0, comm_e2w, &requests[request_index]);
        }
    }

    MPI_Waitall(nWorkers, requests, MPI_STATUSES_IGNORE);

    for (int i = 0; i < nWorkers; ++i) {
        MPI_Ssend(NULL, 0, elem_datatype, i, 0, comm_e2w);
    }

    elem_free(elem);
}

void worker_handler(streamComputeFun stream_compute,
                    MPI_Datatype elem_datatype,
                    streamElemAllocFun elem_alloc,
                    streamElemFreeFun elem_free,
                    int nWorkers,
                    MPI_Comm comm_e2w,
                    MPI_Comm comm_w2c)
{
    int emitter_rank = nWorkers;
    int collector_rank = nWorkers;

    MPI_Status status;
    int status_count;
    void * elem = elem_alloc();

    while (1) {
        MPI_Recv(elem, 1, elem_datatype, emitter_rank, 0, comm_e2w, &status);
        MPI_Get_count(&status, elem_datatype, &status_count);
        if (status_count == 0) break;
        stream_compute(elem);
        MPI_Ssend(elem, 1, elem_datatype, collector_rank, 0, comm_w2c);
    }

    MPI_Ssend(NULL, 0, elem_datatype, collector_rank, 0, comm_w2c);
    elem_free(elem);
}

void collector_handler(streamFinalizeFun stream_finalize,
                       MPI_Datatype elem_datatype,
                       streamElemAllocFun elem_alloc,
                       streamElemFreeFun elem_free,
                       int nWorkers,
                       MPI_Comm comm_w2c)
{
    size_t count = 0;

    MPI_Status status;
    int status_count;
    int * elem = elem_alloc();

    while (1) {
        MPI_Recv(elem, 1, elem_datatype, MPI_ANY_SOURCE, 0, comm_w2c, &status);
        MPI_Get_count(&status, elem_datatype, &status_count);
        if (status_count == 0) {
            count++;
        } else {
            stream_finalize(elem);
        }

        if (count == nWorkers) break;
    }

    elem_free(elem);
}


void farm_skeleton(streamGenerateFun stream_generate,
                   streamComputeFun stream_compute,
                   streamFinalizeFun stream_finalize,
                   MPI_Datatype in_datatype,
                   streamElemAllocFun in_alloc,
                   streamElemFreeFun in_free,
                   MPI_Datatype out_datatype,
                   streamElemAllocFun out_alloc,
                   streamElemFreeFun out_free,
                   int nWorkers,
                   MPI_Comm comm)
{
    int world_rank;
    int world_size;
    MPI_Comm comm_e2w;
    MPI_Comm comm_w2c;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);
    const int emitter_world_rank = world_size - 2;
    const int collector_world_rank = world_size - 1;

    MPI_Comm_split(MPI_COMM_WORLD,
                   (world_rank < nWorkers) || (world_rank == emitter_world_rank),
                   world_rank,
                   &comm_e2w);

    MPI_Comm_split(MPI_COMM_WORLD,
                   (world_rank < nWorkers) || (world_rank == collector_world_rank),
                   world_rank,
                   &comm_w2c);

    double start_time = MPI_Wtime();

    if (world_rank == emitter_world_rank) {
        emitter_handler_issend(stream_generate, in_datatype, in_alloc, in_free, nWorkers, comm_e2w);
    } else if (world_rank == collector_world_rank) {
        collector_handler(stream_finalize, out_datatype, out_alloc, out_free, nWorkers, comm_w2c);
    } else {
        worker_handler(stream_compute, in_datatype, in_alloc, in_free, nWorkers, comm_e2w, comm_w2c);
    }

    double end_time = MPI_Wtime();

    printf("Farm took %f seconds\n", end_time - start_time);
}


int main(int argc, char** argv)
{

    int compute_type = 0;
    streamComputeFun stream_compute_fun;
    if (argc == 2) {
        compute_type = atoi(argv[1]);
    }

    switch (compute_type) {
        case 0:
            stream_compute_fun = stream_compute;
            break;
        case 1:
            stream_compute_fun = stream_sleep_compute;
            break;
        case 2:
            stream_compute_fun = stream_random_sleep_compute;
            break;
        default:
            stream_compute_fun = stream_compute;
            break;
    }

    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size - 2 < 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = world_size - 2;

    MPI_Datatype in_datatype;
    MPI_Type_contiguous(STREAM_ELEM_SIZE, MPI_INT, &in_datatype);
    MPI_Type_commit(&in_datatype);

    farm_skeleton(stream_generate, stream_compute_fun, stream_finalize,
                  in_datatype, stream_in_alloc, stream_in_free,
                  in_datatype, stream_in_alloc, stream_in_free,
                  n, MPI_COMM_WORLD);

    MPI_Type_free(&in_datatype);

    MPI_Finalize();

    return 0;
}
