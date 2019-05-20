TBBCC	= g++
MPICC	= mpicc

MPIRUN	= mpirun

CXXFLAGS	= -std=c++11 -O3 -Wall -pedantic
LIBTBB		= -ltbb

MATRIX_MUL	= matrix_mul
FARM		= farm_skeleton
KMEANS		= kmeans
MANDEL		= mandel

$(MATRIX_MUL) $(FARM) $(KMEANS): % : %.c
	$(MPICC) $< -o $@

$(MANDEL): % : %.cpp
	$(TBBCC) $(CXXFLAGS) $< -o $@ $(LIBTBB)


$(MATRIX_MUL)_run: $(MATRIX_MUL)
	$(MPIRUN) --oversubscribe -np 9 $< 9

$(FARM)_run: $(FARM)
	$(MPIRUN) --oversubscribe -np 8 $< 10

$(KMEANS)_run: $(KMEANS)
	$(MPIRUN) --oversubscribe -np 3 $< 2 1000 5 10 < ./kmeans.data

$(MANDEL)_run: $(MANDEL)
	./$< 1024 1024 -1.1 -0.9 0.15 0.35 8


.PHONY: clean

clean:
	$(RM) -r *.o *~ *.dSYM $(MATRIX_MUL) $(FARM) $(KMEANS) $(MANDEL)
