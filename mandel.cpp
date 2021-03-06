#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <numeric> 
#include "tbb/tbb.h"


static double REAL_BEGIN = -2.0;
static double REAL_END   =  1.0;
static double IMAG_BEGIN = -1.0;
static double IMAG_END   =  1.0;

#define REAL_SCALE(x)       (REAL_BEGIN + (x) * (REAL_END - REAL_BEGIN))
#define IMAG_SCALE(x)       (IMAG_BEGIN + (x) * (IMAG_END - IMAG_BEGIN))
//#define IMAG_SCALE(x)       (IMAG_END + (x) * (IMAG_BEGIN - IMAG_END))

#define MAX_ITERATIONS      1024
#define MAX_SMODULUS        4.0


struct Stats {
    size_t row_b;
    size_t row_e;
    size_t col_b;
    size_t col_e;
    std::vector<size_t> freqs;

    Stats(size_t row_b,
          size_t row_e,
          size_t col_b,
          size_t col_e) :
    row_b(row_b),
    row_e(row_e),
    col_b(col_b),
    col_e(col_e),
    freqs(MAX_ITERATIONS)
    {}
};


size_t mand_compute(double cx, double cy) {
    size_t i;
    double x = cx;
    double y = cy;
    double nx;
    double ny;

    for (i = 1; i < MAX_ITERATIONS; ++i) {
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

class Mandel {
    std::vector<size_t> & out;
    tbb::concurrent_vector<Stats> & q;
    const size_t rows;
    const size_t cols;

public:
    void operator()(const tbb::blocked_range2d<size_t> & r) const {
        const size_t rs = rows;
        const size_t cs = cols;

        Stats local_stats(r.rows().begin(),
                          r.rows().end(),
                          r.cols().begin(),
                          r.cols().end());

        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                const size_t index = i * cs + j;
                const double cx = REAL_SCALE((double)j / cs);
                const double cy = IMAG_SCALE((double)i / rs);
                const size_t iterations = mand_compute(cx, cy);
                out[index] = iterations;
                local_stats.freqs[iterations]++;
            }
        }

        q.push_back(local_stats);
    }

    Mandel(std::vector<size_t> & out,
           tbb::concurrent_vector<Stats> & q,
           const size_t rows,
           const size_t cols) :
    out(out),
    q(q),
    rows(rows),
    cols(cols)
    {}
};


// C routine to save a 2d int array as an image to a simple graphic format
// edited form of code from the rosetta code project
// https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
//
// filename must end with ".ppm"
int save_image(const char * filename,
               const std::vector<size_t> & matrix, const size_t rows, const size_t cols,
               const size_t max_value)
{
    assert(filename != NULL);
    assert(rows > 0);
    assert(cols > 0);
    assert(max_value > 0);

    FILE * fp = fopen(filename, "wb");
    if (fp == NULL) {
        return -1;
    }

    fprintf(fp, "P6\n%zu %zu\n255\n", rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const size_t val = matrix[i * cols + j];
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

size_t max_array(const std::vector<size_t> & array, const size_t size)
{
    size_t max = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < size; ++i) {
        const size_t val = array[i];
        max = (max > val ? max : val);
    }
    return max;
}

double weighted_average(std::vector<size_t> & v)
{
    size_t iterations = 0;
    size_t sum = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        iterations += v[i];
        sum += v[i] * i;
    }

    return (double)sum / iterations;
}

int main(int argc, char * argv[])
{
    char * filename = NULL;
    // to avoid overflow when user specifies negative values is used ssize_t == signed size_t
    ssize_t rows = (1 << 10) * 2;
    ssize_t cols = (1 << 10) * 3;
    ssize_t threads = 4;
    ssize_t row_grain = 0;
    ssize_t col_grain = 0;

    if (argc < 2) {
        std::cerr << "usage: ./mandel filename.ppm [rows cols] [real_start real_end imag_start imag_end] [threads] [row_grain col_grain]" << std::endl;
        exit(-1);
    }

    argc--;
    argv++;
    size_t argi = 0;
    if (argc > 0) filename   = argv[argi++];
    
    // (argc > 2) to take both rows and cols
    if (argc > 2) rows       = strtol(argv[argi++], NULL, 10);
    if (argc > 2) cols       = strtol(argv[argi++], NULL, 10);
    
    // (argc > 6) to take both starts and ends values of REAL and IMAG ranges
    if (argc > 6) REAL_BEGIN = strtod(argv[argi++], NULL);
    if (argc > 6) REAL_END   = strtod(argv[argi++], NULL);
    if (argc > 6) IMAG_BEGIN = strtod(argv[argi++], NULL);
    if (argc > 6) IMAG_END   = strtod(argv[argi++], NULL);

    if (argc > 7) threads    = strtol(argv[argi++], NULL, 10);

    // (argc > 9) to take row and col grain sizes
    if (argc > 9) row_grain  = strtol(argv[argi++], NULL, 10); 
    if (argc > 9) col_grain  = strtol(argv[argi++], NULL, 10);


    assert(rows > 0);
    assert(cols > 0);
    assert(threads > 0);
    assert(row_grain >= 0);
    assert(col_grain >= 0);

    std::cout << "     rows: "  << rows       << std::endl;
    std::cout << "     cols: "  << cols       << std::endl;
    std::cout << "     REAL: [" << REAL_BEGIN << ", "<< REAL_END << "]" << std::endl;
    std::cout << "     IMAG: [" << IMAG_BEGIN << ", "<< IMAG_END << "]" << std::endl;
    std::cout << "  threads: "  << threads    << std::endl;
    if (row_grain > 0) std::cout << "row_grain: " << row_grain << std::endl;
    if (col_grain > 0) std::cout << "col_grain: " << col_grain << std::endl;

    // Preparing parallel computation
    // Data structures
    std::vector<size_t> out(rows * cols);
    tbb::concurrent_vector<Stats> q;
    Mandel m(out, q, rows, cols);

    // Settings the number of threads
    tbb::task_scheduler_init init(threads);

    // Grain sizes
    tbb::blocked_range2d<size_t> range = tbb::blocked_range2d<size_t>(0, rows, 0, cols);
    if (row_grain > 0 && col_grain > 0) {
        range = tbb::blocked_range2d<size_t>(0, rows, row_grain, 0, cols, col_grain);
    }

    // Computation
    tbb::tick_count time_start = tbb::tick_count::now();
    parallel_for(range, m);
    tbb::tick_count time_stop = tbb::tick_count::now();


    // Statistics
    size_t w_range = 4;
    size_t width = 24;
    std::cout << std::setw(width) << "Range";
    std::cout << std::setw(width) << "Elements";
    std::cout << std::setw(width) << "MinIter";
    std::cout << std::setw(width) << "MinFreq";
    std::cout << std::setw(width) << "Avg";
    std::cout << std::setw(width) << "MaxIter";
    std::cout << std::setw(width) << "MaxFreq";
    std::cout << std::endl;

    for (Stats & s : q) {
        size_t elements = (s.row_e - s.row_b) * (s.col_e - s.col_b);
        auto minmax = std::minmax_element(s.freqs.begin(), s.freqs.end());
        double freq_avg = weighted_average(s.freqs);

        std::cout << "[";
        std::cout << std::setw(w_range) << s.row_b << ", ";
        std::cout << std::setw(w_range) << s.row_e << ", ";
        std::cout << std::setw(w_range) << s.col_b << ", ";
        std::cout << std::setw(w_range) << s.col_e << "]";

        std::cout << std::setw(width) << elements;
        std::cout << std::setw(width) << minmax.first - s.freqs.begin();
        std::cout << std::setw(width) << *minmax.first;
        std::cout << std::setw(width) << freq_avg;
        std::cout << std::setw(width) << minmax.second - s.freqs.begin();
        std::cout << std::setw(width) << *minmax.second;

        std::cout << std::endl;
    }

    // Timing
    double time_s = (time_stop - time_start).seconds();
    std::cout << " timing: " << time_s << " s" << std::endl;

    size_t max_value = max_array(out, rows * cols);
    save_image(filename, out, cols, rows, max_value);

    return 0;
}