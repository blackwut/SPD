// k = 4
// l = 3

// 1 + 4 + 4^2 = 21 (4^(3 - 1) + 4 + 1)
// 1 + 4 + 4^2 + 4^3 = (4^(4 - 1) + 4 + 1)  

//                                                                               1
// |                   2                   |                   2                   |                   2                   |                   2                   | 
// |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    |    3    | 
// | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 

// | 1 |
// | 2 2 2 2 | 
// | 3 3 3 3 | 3 3 3 3 | 3 3 3 3 | 3 3 3 3 | 
// | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 

//   0 1 2 3   4 5 6 7 
// | 1                                                                                                                                                             | 0
// | 2                                       2                                       2                                       2                                     | 1
// | 3         3         3         3       | 3         3         3         3       | 3         3         3         3       | 3         3         3         3       | 2
// | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 4 4 4 4 | 3



// comm_rank / k ^ 3 -> group0             if group0.rank == 0 -> 1
// comm_rank / k ^ 2 -> group1             if group1.rank == 0 -> 2 2 2 2
// comm_rank / k ^ 1 -> group2             if group2.rank == 0 -> 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3



// MPI_Comm_split(MPI_COMM_WORLD, world_rank / k, world_rank, &foglie); // Foglie in gruppi da 4

// if (foglie_rank == 0) {
//     MPI_Comm_split(MPI_COMM_WORLD, world_rank / k, world_rank, &foglie_1); // Foglie - 1 in gruppi da 4
// } else {
//     MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, world_rank, &foglie_1);
// }



// for (size_t i = l; i > 0; --i) {
//     MPI_Comm_split(comm, rank / k, rank, &newcomm);
// }

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ELEM_SIZE 16

int pow_int(int x, int y)
{
    int res = 1;
    while (y--) res *= x;
    return res;
}

void ktree(int k, int l, MPI_Datatype elem_datatype, MPI_Comm comm)
{
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    MPI_Comm * levels = (MPI_Comm *)calloc(l + 1, sizeof(MPI_Comm));
    int * ranks = (int *)calloc(l + 1, sizeof(int));
    int * sizes = (int *)calloc(l + 1, sizeof(int));


    levels[0] = comm;
    ranks[0] = comm_rank;
    sizes[0] = comm_size;

    for (int i = 0; i < l - 1; ++i) {
        int size = pow_int(k, l - i);
        int color = comm_rank / size;
        // printf("color: %2d - size: %2d\n", color, size);
        color = comm_rank < size ? color : MPI_UNDEFINED;
        MPI_Comm_split(comm, color, comm_rank, &levels[i]);

        if (levels[i] != MPI_COMM_NULL) {
            MPI_Comm_rank(levels[i], &ranks[i]);
            MPI_Comm_size(levels[i], &sizes[i]);

             for (int j = 0; j < sizes[i]; ++j) {
                if (ranks[i] == j) {
                    printf("level: %2d - comm_rank: %2d - level_rank: %2d\n", i, comm_rank, ranks[i]);
                    fflush(stdout);
                }
                MPI_Barrier(levels[i]);
            }
        }
    }

    // MPI_Comm_split(comm, comm_rank / k, comm_rank, &levels[l - 1]);
    // for (int i = l - 1; i > 0; ++i) {

    //     if (levels[i] != MPI_COMM_NULL) {
    //         MPI_Comm_rank(levels[i], &ranks[i]);
    //         MPI_Comm_size(levels[i], &sizes[i]);

    //         for (int j = 0; j < sizes[i]; ++j) {
    //             if (ranks[i] == j) {
    //                 printf("level: %d - comm_rank: %d - level_rank: %d\n", i, comm_rank, ranks[i]);
    //                 fflush(stdout);
    //             }
    //             MPI_Barrier(levels[i]);
    //         }

    //         int color = ranks[i] == 0 ? comm_rank / pow_int(k, l - i) : MPI_UNDEFINED;
    //         MPI_Comm_split(levels[i], comm_rank / k, comm_rank, &levels[i - 1]);
    //     }
    // }

    // for (int j = 0; j < comm_size; ++j) {
    //     if (comm_rank == j) {
    //         printf("%d) %d\n", l - 1, j);
    //         fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // int level_nodes = pow_int(k, l);
    // int color = comm_rank / level_nodes;
    // MPI_Comm_split(comm, color, comm_rank, &levels[l - 1]);

    // if (levels[l - 1] != MPI_COMM_NULL) {
    //     MPI_Comm_rank(levels[l - 1], &ranks[l - 1]);
    //     MPI_Comm_size(levels[l - 1], &sizes[l - 1]);
    // }

    // for (int j = 0; j < comm_size; ++j) {
    //     if (comm_rank == j) {
    //         printf("level: %d - comm_rank: %d - level_rank: %d\n", l - 1, comm_rank, ranks[l - 1]);
    //         fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // level_nodes = pow_int(k, l - 1);
    // MPI_Comm_split(comm, comm_rank / level_nodes, comm_rank, &levels[l - 2]);
    // if (levels[l - 2] != MPI_COMM_NULL) {
    //     MPI_Comm_rank(levels[l - 2], &ranks[l - 2]);
    //     MPI_Comm_size(levels[l - 2], &sizes[l - 2]);
    // }

    // for (int j = 0; j < comm_size; ++j) {
    //     if (comm_rank == j) {
    //         printf("level: %d - comm_rank: %d - level_rank: %d\n", l - 2, comm_rank, ranks[l - 2]);
    //         fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // level_nodes = pow_int(k, l - 2);
    // MPI_Comm_split(comm, comm_rank / level_nodes, comm_rank, &levels[l - 3]);
    // if (levels[l - 3] != MPI_COMM_NULL) {
    //     MPI_Comm_rank(levels[l - 1], &ranks[l - 1]);
    //     MPI_Comm_size(levels[l - 1], &sizes[l - 1]);
    // }

    // for (int j = 0; j < comm_size; ++j) {
    //     if (comm_rank == j) {
    //         printf("level: %d - comm_rank: %d - level_rank: %d\n", l - 3, comm_rank, ranks[l - 3]);
    //         fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // level_nodes = pow_int(k, l - 3);
    // MPI_Comm_split(comm, comm_rank / level_nodes, comm_rank, &levels[l - 4]);
    // if (levels[l - 3] != MPI_COMM_NULL) {
    //     MPI_Comm_rank(levels[l - 4], &ranks[l - 4]);
    //     MPI_Comm_size(levels[l - 4], &sizes[l - 4]);
    // }

    // for (int j = 0; j < comm_size; ++j) {
    //     if (comm_rank == j) {
    //         printf("level: %d - comm_rank: %d - level_rank: %d\n", l - 4, comm_rank, ranks[l - 4]);
    //         fflush(stdout);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }

    // int level_nodes = k;
    // MPI_Comm_split(comm, comm_rank / level_nodes, comm_rank, &levels[l - 1]);
    // for (int i = l - 1; i >= 0; --i) {

    //     if (levels[i] != MPI_COMM_NULL) {
    //         MPI_Comm_rank(levels[i], &ranks[i]);
    //         MPI_Comm_size(levels[i], &sizes[i]);
    //     }

    //     for (int j = 0; j < comm_size; ++j) {
    //         if (comm_rank == j) {
    //             printf("level: %d - comm_rank: %d - level_rank: %d\n", i, comm_rank, ranks[i]);
    //             fflush(stdout);
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);
    //     }

    //     level_nodes *= k;
    //     int color = (ranks[i] == 0 ? comm_rank / level_nodes : MPI_UNDEFINED);
    //     MPI_Comm_split(comm, color, comm_rank, &levels[i - 1]);
    // }
}



int main(int argc, char** argv)
{

    if (argc < 2) {
        fprintf(stderr, "Please provide the arity of the tree!\n");
        exit(-1);
    }
    if (argc < 3) {
        fprintf(stderr, "Please provide how many levels does have the tree!\n");
        exit(-1);
    }

    int k = atoi(argv[1]);
    int l = atoi(argv[2]);

    int world_rank;
    int world_size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Datatype elem_datatype;
    MPI_Type_contiguous(ELEM_SIZE, MPI_INT, &elem_datatype);
    MPI_Type_commit(&elem_datatype);

    // printf("pow %d ^ %d = %d\n", k, l, pow_int(k, l));
    if (pow_int(k, l - 1) > world_size) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ktree(k, l, elem_datatype, MPI_COMM_WORLD);

    MPI_Type_free(&elem_datatype);
    MPI_Finalize();

    return 0;
}
