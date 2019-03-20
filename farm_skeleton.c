#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <unistd.h>

#define EOS       -1
#define STREAM_ELEM_SIZE 10
#define MAX_STREAM_COUNT 10


#define STREAM_ELEM_EOS 3

typedef int (*streamGenerateFun)(void *);
typedef void (*streamComputeFun)(void *);
typedef void (*streamFinalizeFun)(void *);
typedef void * (*streamElemAllocFun)();
typedef void (*streamElemFreeFun)(void *);


void * stream_elem_alloc()
{
    return calloc(STREAM_ELEM_SIZE, sizeof(int));
}

void stream_elem_free(void * stream_elem)
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

void stream_finalize(void * stream_elem)
{
    int * elem = (int *)stream_elem;
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
        MPI_Send(elem, 1, elem_datatype, i, 0, comm_e2w);
        i = (i + 1) % nWorkers;
    }

    for (i = 0; i < nWorkers; ++i) {
        MPI_Send(NULL, 0, elem_datatype, i, 0, comm_e2w);
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
        MPI_Send(elem, 1, elem_datatype, collector_rank, 0, comm_w2c);
    }

    MPI_Send(NULL, 0, elem_datatype, collector_rank, 0, comm_w2c);
}

void collector_handler(streamFinalizeFun stream_finalize,
                       MPI_Datatype elem_datatype,
                       streamElemAllocFun elem_alloc,
                       streamElemFreeFun elem_free,
                       int nWorkers,
                       MPI_Comm comm_w2c)
{
    int i = 0;
    int count = 0;

    MPI_Status status;
    int status_count;
    int * elem = elem_alloc();

    while (1) {
        MPI_Recv(elem, 1, elem_datatype, i, 0, comm_w2c, &status);
        MPI_Get_count(&status, elem_datatype, &status_count);
        if (status_count == 0) count++;
        if (count == nWorkers) break;
        

        stream_finalize(elem);
        i = (i + 1) % nWorkers;
    }

    elem_free(elem);
}


void farm_skeleton(streamGenerateFun stream_generate,
                   streamComputeFun stream_compute,
                   streamFinalizeFun stream_finalize,
                   MPI_Datatype elem_datatype,
                   streamElemAllocFun elem_alloc,
                   streamElemFreeFun elem_free,
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

    if (world_rank == emitter_world_rank) {
        emitter_handler(stream_generate, elem_datatype, elem_alloc, elem_free, nWorkers, comm_e2w);
    } else if (world_rank == collector_world_rank) {
        collector_handler(stream_finalize, elem_datatype, elem_alloc, elem_free, nWorkers, comm_w2c);
    } else {
        worker_handler(stream_compute, elem_datatype, elem_alloc, elem_free, nWorkers, comm_e2w, comm_w2c);
    }
}


int main(int argc, char** argv)
{
    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size - 2 < 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n = world_size - 2;

    MPI_Datatype elem_datatype;
    MPI_Type_contiguous(STREAM_ELEM_SIZE, MPI_INT, &elem_datatype);
    MPI_Type_commit(&elem_datatype);

    farm_skeleton(stream_generate, stream_compute, stream_finalize,
                  elem_datatype, stream_elem_alloc, stream_elem_free,
                  n, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
