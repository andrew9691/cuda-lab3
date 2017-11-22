#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

__global__ void turnmat(uchar *image, uchar *out_image, int rows, int cols)
{
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= rows || j >= cols)
        return;

    int out_i = cols - 1 - j;
    int out_j = i;
    ((uchar3*)out_image)[out_i * rows + out_j] = ((uchar3*)image)[i * cols + j];
}

#define shared_x 16
#define shared_y 16

__global__ void shared_turnmat(uchar *image, uchar *out_image, int rows, int cols)
{
    __shared__ uchar4 temp[shared_x][shared_y+1]; // blockDim.x * blockDim.y
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= rows || j >= cols)
        return;

    int new_i = shared_x - 1 - tx;
    int new_j = ty;
    uchar3 ttt = ((uchar3*)image)[i * cols + j];
    temp[new_i][new_j] = make_uchar4(ttt.x, ttt.y, ttt.z,0);

     __syncthreads();

    int out_i = cols - 1 - blockIdx.x * blockDim.x + ty;
    int out_j = blockIdx.y * blockDim.y + tx;
    uchar4 tttt = temp[ty][tx];
    ((uchar3*)out_image)[out_i * rows + out_j] = make_uchar3(tttt.x, tttt.y, tttt.z);
}

int main(void)
{
    Mat image;

    image = imread("pic.jpeg", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! image.data )                              // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat out_image1(image.cols, image.rows, DataType<Vec3b>::type);

    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    startCPU = clock();
#define tile_i 32
#define tile_j 16
    for (int it = 0; it < image.rows/tile_i*tile_i; it+=tile_i)
    {
        for (int jt = 0; jt < image.cols/tile_j*tile_j; jt+=tile_j)
        {
            for(int ii = 0; ii<tile_i; ii++)
            for(int jj = 0; jj<tile_j; jj++)
            {
              int i = it + ii, j = jt+jj;
              ((Vec3b*)out_image1.data)[out_image1.cols*j+i] = ((Vec3b*)image.data)[image.cols*i + image.cols - j - 1];
            }
        }
    }

    elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
    cout << "CPU memory throughput = " << 6*image.cols*image.rows/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

    imwrite("pic_resCPU.jpeg", out_image1);

////////////////////////////////////////////////////////////////////////////////////////////////////

    Mat out_image(image.cols, image.rows, DataType<Vec3b>::type);
    uchar *dev_src_image;

    uchar * res_src_image;
    CHECK( cudaMalloc(&res_src_image, 3 * out_image.cols * out_image.rows) );
    CHECK( cudaMemcpy(res_src_image, out_image.data, 3 * out_image.cols * out_image.rows, cudaMemcpyHostToDevice) );

    CHECK( cudaMalloc(&dev_src_image, 3 * image.cols * image.rows) );
    CHECK( cudaMemcpy(dev_src_image, image.data, 3 * image.cols * image.rows, cudaMemcpyHostToDevice) );

    cudaEventRecord(startCUDA,0);

    //int bx = 4, by = 32;
    int bx = shared_x, by = shared_y;
    //turnmat<<<dim3((image.cols + (bx-1)) / bx, (image.rows + (by-1)) / by, 1), dim3(bx, by, 1)>>>(dev_src_image, res_src_image, image.rows, image.cols);
    shared_turnmat<<<dim3((image.cols + (bx-1)) / bx, (image.rows + (by-1)) / by, 1), dim3(bx, by, 1)>>>(dev_src_image, res_src_image, image.rows, image.cols);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 6*image.cols*image.rows/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";
    CHECK(cudaMemcpy(out_image.data, res_src_image, 3 * image.cols * image.rows, cudaMemcpyDeviceToHost));

    imwrite("pic_resGPU.jpeg", out_image);
    return 0;

////////////////////////////////////////////////////////////////////////////////////////////////////
}
