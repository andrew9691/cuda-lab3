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

__global__ void turnmat(uchar *image, uchar *out_image, int cols, int rows)
{
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    int j = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= rows || j >= cols)
        return;

    uchar *p = image + 3 * (i * cols + j);
    uchar *out_p = out_image + 3 * (i * rows + (cols - 1 - j)); // что-то не так

    for (int ch = 0; ch < 3; ch++)
    {
      *(out_p + ch) = *(p + ch);
    }
}

int main(void)
{
    int N = 10*1000*1000;
    Mat image;

    image = imread("pic.jpeg", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat out_image(image.cols, image.rows, DataType<Vec3b>::type);

    cudaEvent_t startCUDA, stopCUDA;
    clock_t startCPU;
    float elapsedTimeCUDA, elapsedTimeCPU;
    cudaEventCreate(&startCUDA);
    cudaEventCreate(&stopCUDA);

    startCPU = clock();

    for (int i = 0; i < image.rows; i++)
    {
        Vec3b* p = image.ptr<Vec3b>(i);
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b* out_p = out_image.ptr<Vec3b>(j);
            out_p[i] = p[image.cols - j - 1];
        }
    }

    elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
    cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
    cout << "CPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

////////////////////////////////////////////////////////////////////////////////////////////////////

    uchar *dev_src_image;

    uchar * res_src_image;
    CHECK( cudaMalloc(&res_src_image, 3 * out_image.cols * out_image.rows) );
    CHECK( cudaMemcpy(res_src_image, out_image.data, 3 * out_image.cols * out_image.rows, cudaMemcpyHostToDevice) );

    CHECK( cudaMalloc(&dev_src_image, 3 * image.cols * image.rows) );
    CHECK( cudaMemcpy(dev_src_image, image.data, 3 * image.cols * image.rows, cudaMemcpyHostToDevice) );

    cudaEventRecord(startCUDA,0);

    // может поменять gridDim и blockDim местами?
    turnmat<<<dim3((image.rows + 15) / 16, (image.cols + 15) / 16, 1), dim3(16, 16, 1)>>>(dev_src_image, res_src_image, image.cols, image.rows);

    cudaEventRecord(stopCUDA,0);
    cudaEventSynchronize(stopCUDA);
    CHECK(cudaGetLastError());

    cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

    //out_image = out_image.clone();

    cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
    cout << "CUDA memory throughput = " << 3*N*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";
    CHECK(cudaMemcpy(out_image.data, dev_src_image, 3 * image.cols * image.rows, cudaMemcpyDeviceToHost));

////////////////////////////////////////////////////////////////////////////////////////////////////

    imwrite("pic_res.jpeg", out_image);
    return 0;
}
