#include <iostream>
#include "tbb/tbb.h"

using namespace tbb;

class ApplyFoo {
    size_t n;
    float * my_a;
    float * my_b;

public:
    void operator()(const blocked_range<size_t> & r) const {
        float * a = my_a;
        float * b = my_b;
        for (size_t i = r.begin(); i != r.end(); ++i) {
            b[i] = a[i] * a[i];
        }
    }
    
    ApplyFoo(const size_t n) : n(n) {
        my_a = new float[n];
        my_b = new float[n];

        for (size_t i = 0; i < n; ++i) {
            my_a[i] = i;
            my_b[i] = 0;
        }
    }

    void print() {
        for (size_t i = 0; i < n; ++i) {
            std::cout << my_b[i] << " ";
        }
        std::cout << std::endl;
    }
};

void ParallelApplyFoo(size_t n)
{
    ApplyFoo foo(n);
    parallel_for(blocked_range<size_t>(0, n), foo);
    foo.print();
}

class ApplyFoo2d {
    size_t n;
    size_t m;
    float * my_a;
    float * my_b;

public:
    void operator()(const blocked_range2d<size_t> & r) const {
        float * a = my_a;
        float * b = my_b;
        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                b[i * n + j] = a[i * n + j] * a[i * n + j];
            }
        }
    }
    
    ApplyFoo2d(const size_t n, const size_t m) : n(n), m(m) {
        my_a = new float[n * m];
        my_b = new float[n * m];

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                my_a[i * n + j] = i * n + j;
                my_b[i * n + j] = 0;
            }
        }
    }

    void print() {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                std::cout << my_b[i] << " ";
            }
            std::cout << std::endl;
        }
    }
};

void ParallelApplyFoo2d(const size_t n, const size_t m)
{
    ApplyFoo2d foo(n, m);
    parallel_for(blocked_range2d<size_t>(0, n, 0, m), foo);
    foo.print();
}

int main (int argc, char * argv[])
{
    size_t n = 1 << 4;
    size_t m = 1 << 4;
    ParallelApplyFoo(n);
    ParallelApplyFoo2d(n, m);

    return 0;
}