#include <iostream>
#include <ctime>
#include <omp.h>
#include <cstdlib>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
// #include <chrono>

using namespace std;
//using namespace cv;

int main( int argc, char** argv )
{
  double start = omp_get_wtime();

  int cols = 2560;
  int rows = 1600;
  char mat[cols*rows];
  char out_mat[cols*rows];
  #define tile_i 32
  #define tile_j 16

  // typedef std::chrono::high_resolution_clock Time;
  // auto start_time = Time::now();
  #pragma omp parallel for
  for (int it = 0; it < rows/tile_i*tile_i; it+=tile_i)
  {
     for (int jt = 0; jt < cols/tile_j*tile_j; jt+=tile_j)
    {
        for(int ii = 0; ii<tile_i; ii++)
        for(int jj = 0; jj<tile_j; jj++)
          {
            int i = it + ii, j = jt+jj;
            out_mat[rows * j + i] = mat[cols * i + cols - j - 1];
          }
      }
  }

  // cout << "Time: " << (std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - start_time)).count() << "ms" << endl;
  double elapsedTimeCPU = omp_get_wtime() - start;
  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";

}
