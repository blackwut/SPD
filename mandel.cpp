#include <iostream>
#include <string>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "tbb/tbb.h"

using namespace tbb;
using namespace std;

static double REAL_START = -2.0;
static double REAL_END   =  1.0;
static double IMAG_START = -1.0;
static double IMAG_END   =  1.0;

#define REAL_SCALE(x)       (REAL_START + (x) * (REAL_END - REAL_START))
#define IMAG_SCALE(x)       (IMAG_START + (x) * (IMAG_END - IMAG_START))

#define DEFAULT_ITERATIONS  512
#define MAX_SMODULUS        4.0


class Mandel {
    int * output;
    const size_t rows;
    const size_t cols;

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

            if (nx * nx + ny * ny > MAX_SMODULUS) {
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
            const int val = matrix[i * cols + j];
            static unsigned char color[3];
            color[0] = (max_value != 255 ? val / max_value * 255 : val);
            color[1] = val % 256;
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
    size_t rows = (1 << 10) * 2;
    size_t cols = (1 << 10) * 3;
    size_t threads = 4;

    if (argc > 0) rows       = strtol(argv[0], NULL, 10);
    if (argc > 1) cols       = strtol(argv[1], NULL, 10);
    if (argc > 2) REAL_START = strtod(argv[2], NULL);
    if (argc > 3) REAL_END   = strtod(argv[3], NULL);
    if (argc > 4) IMAG_START = strtod(argv[4], NULL);
    if (argc > 5) IMAG_END   = strtod(argv[5], NULL);
    if (argc > 6) threads    = strtol(argv[6], NULL, 10);

    std::cout << "   rows: "  << rows       << std::endl;
    std::cout << "   cols: "  << cols       << std::endl;
    std::cout << "   REAL: [" << REAL_START << ", "<< REAL_END << "]" << std::endl;
    std::cout << "   IMAG: [" << IMAG_START << ", "<< IMAG_END << "]" << std::endl;
    std::cout << "threads: "  << threads    << std::endl;

    // Preparing parallel computation
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