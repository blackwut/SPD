#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <assert.h>
#include "tbb/tbb.h"

using namespace tbb;
using namespace std;

#define DEFAULT_X -2.0
#define DEFAULT_Y -2.0
#define DEFAULT_ITERATIONS 1024
#define MAX_SMODULUS 4


class Mandel {
    const size_t rows;
    const size_t cols;
    int * output;

public:
    void operator()(const blocked_range2d<size_t> & r) const
    {
        const size_t c = cols;
        const size_t f = (rows > cols ? rows : cols) / 4;
        int * out = output;

        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                const double cx = (double)j / f + DEFAULT_X;
                const double cy = (double)i / f + DEFAULT_Y;
                out[i * c + j] = mand_compute(cx, cy);
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
    int mand_compute(double cx, double cy) const
    {
        int i;
        double x = cx;
        double y = cy;
        double nx;
        double ny;

        for (i = 0; i < DEFAULT_ITERATIONS; ++i) {
            // (x, y)^2 + c
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
    size_t rows = 1 << 12;
    size_t cols = 1 << 13;

    int * output = new int[rows * cols];

    Mandel m(output, rows, cols);
    parallel_for(blocked_range2d<size_t>(0, rows, 0, cols), m);
    int max_value = max_array(output, rows * cols);
    save_image("/Volumes/RamDisk/mandel.ppm", output, cols, rows, max_value);

    return 0;
}