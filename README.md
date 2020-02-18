# SPD

SPD exercises repository

## How to run exercises with Makefile

With make file you can run all the exercises simpli append "_run" to the chosen target.
Targets are: "matrix_mul", "farm_skeleton", "kmeans", "mandel".
For example:
```bash
make farm_skeleton_run
```
The result will be:
```bash
mpicc -O3 matrix_mul.c -o matrix_mul
mpirun --oversubscribe -np 9 matrix_mul 9
C
   4    8   12    4    8   12    4    8   12 
  40   44   48   40   44   48   40   44   48 
  76   80   84   76   80   84   76   80   84 
 124  128  132  124  128  132  124  128  132 
 160  164  168  160  164  168  160  164  168 
 196  200  204  196  200  204  196  200  204 
 244  248  252  244  248  252  244  248  252 
 280  284  288  280  284  288  280  284  288 
 316  320  324  316  320  324  316  320  324 
```


## And without Makefile

### ex2.c

```bash
mpicc -O3 ex2.c -o ex2 && mpirun -np 2 ex2
```

### matrix_mul.c

Number of processors `P` must be a square number and must be a divider of `N` (the size of square matrices computed).

```bash
mpicc -O3 matrix_mul.c -o matrix_mul && mpirun --oversubscribe -np P matrix_mul N
```

### mandel.cpp
```bash
g++ -std=c++11 -O3 mandel.cpp -o mandel -ltbb && ./mandel 1024 1024 -1.1 -0.9 0.15 0.35 8
```





