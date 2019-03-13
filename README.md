# SPD

SPD exercises repository

## How to run exercises

### ex2.c

```bash
mpicc ex2.c -o ex2 && mpirun -np 2 ex2
```

### matrix_mul.c

Number of processors `P` must be a square number and must be a divider of `N` (the size of square matrices computed).

```bash
mpicc matrix_mul.c -o matrix_mul && mpirun --oversubscribe -np P matrix_mul N
```