#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "tbb/tbb.h"

using namespace tbb;
using namespace std;

#define REAL_START              -2.0
#define REAL_END                 1.0
#define IMAG_START              -1.0
#define IMAG_END                 1.0
#define REAL_SCALE(x)           (REAL_START + (x) * (REAL_END - REAL_START))
#define IMAG_SCALE(x)           (IMAG_START + (x) * (IMAG_END - IMAG_START))

#define DEFAULT_ITERATIONS      64
#define MAX_SMODULUS             4.0


class Mandel {
    const size_t rows;
    const size_t cols;
    int * output;

public:
    void operator()(const blocked_range2d<size_t> & r) const {
        const size_t rs = rows;
        const size_t cs = cols;
        int * out = output;

        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                const double cx = REAL_SCALE((double)j / cs);
                const double cy = IMAG_SCALE((double)i / rs);
                out[i * cs + j] = mand_compute(cx, cy);
            }
        }
    }
    
    Mandel(int * output, const size_t rows, const size_t cols)
    : output(output), rows(rows), cols(cols)
    {}

    void print() {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::cout << output[i] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    //returns the number of iteration until divergence of a single point
    int mand_compute(double cx, double cy) const {
        int i;
        double x = cx;
        double y = cy;
        double nx;
        double ny;

        for (i = 1; i < DEFAULT_ITERATIONS; ++i) {
            nx = x * x - y * y + cx;
            ny = 2 * x * y + cy;

            if (fabs(nx * nx + ny * ny) > MAX_SMODULUS) {
                break;
            }
            x = nx;
            y = ny;
        }

        return i;
    }
};


// C routine to save a 2d int array as an image to a simple graphic format
// edited form of code from the rosetta code project
// https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
//
// filename must end with ".ppm"
int save_image(const char * filename,
               const int * matrix, const size_t rows, const size_t cols,
               const int max_value)
{
    assert(filename != NULL);
    assert(matrix != NULL);
    assert(rows > 0);
    assert(cols > 0);
    assert(max_value > 0);

    FILE * fp = fopen(filename, "wb");
    if (fp == NULL) {
        return -1;
    }

    fprintf(fp, "P6\n%zu %zu\n255\n", rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int val = matrix[i * cols + j];
            if (max_value != 255) {
                val = (val * 255) / max_value;
            }
            static unsigned char color[3];
            color[0] = val % 256;
            color[1] = 0;
            color[2] = 0;
            fwrite(color, 1, 3, fp);
        }
    }

    fclose(fp);

    return 0;
}

int max_array(const int * array, const size_t size)
{
    int max = INT_MIN;
    for (size_t i = 0; i < size; ++i) {
        const int val = array[i];
        max = (max > val ? max : val);
    }
    return max;
}

int main(int argc, char * argv[])
{
    argc--;
    argv++;
    size_t rows = 1 << 10;
    size_t cols = 1 << 10;
    size_t threads = 1;

    if (argc > 0) rows    = 1 << strtol(argv[0], NULL, 10);
    if (argc > 1) cols    = 1 << strtol(argv[1], NULL, 10);
    if (argc > 2) threads = strtol(argv[2], NULL, 10);

    rows *= 2;
    cols *= 3;

    std::cout << "   rows: " << rows    << std::endl;
    std::cout << "   cols: " << cols    << std::endl;
    std::cout << "threads: " << threads << std::endl;

    // Preparing paralle computation
    int * output = new int[rows * cols];
    Mandel m(output, rows, cols);

    // Executing parallel computation
    tbb::task_scheduler_init init(threads);
    tbb::tick_count time_start = tbb::tick_count::now();
    parallel_for(blocked_range2d<size_t>(0, rows, 0, cols), m);
    tbb::tick_count time_stop = tbb::tick_count::now();

    double time_s = (time_stop - time_start).seconds();
    std::cout << " timing: " << time_s << " s" << std::endl;

    int max_value = max_array(output, rows * cols);
    save_image("/Volumes/RamDisk/mandel.ppm", output, cols, rows, max_value);

    delete[] output;

    return 0;
}