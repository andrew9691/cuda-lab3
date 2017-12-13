#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;
using namespace cv;

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    cout << "Error in " #func "\nError code = " << errcode << "\n";\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);

// __kernel void turnmat(__global uchar* in_img, __global uchar* out_img, int rows, int cols)
// {
//     int i = get_global_id(1);
//     int j = get_global_id(0);
//     if (i >= rows || j >= cols)
//         return;
//
//     int out_i = cols - 1 - j;
//     int out_j = i;
//
//     ((uchar3*)out_img)[out_i * rows + out_j] = ((uchar3*)in_img)[i * cols + j];
// }

#define shared_x 16
#define shared_y 16

// __kernel void shared_turnmat(uchar *image, uchar *out_image, int rows, int cols)
// {
//     __local uchar4 temp[shared_x][shared_y + 1];
//     int tx = get_local_id(0);
//     int ty = get_local_id(1);
//     int i = get_global_id(1);
//     int j = get_global_id(0);
//     if (i >= rows || j >= cols)
//         return;
//
//     int new_i = shared_x - 1 - tx;
//     int new_j = ty;
//     uchar3 ttt = ((uchar3*)image)[i * cols + j];
//     temp[new_i][new_j] = make_uchar4(ttt.x, ttt.y, ttt.z, 0);
//
//     barrier();
//
//     int out_i = cols - 1 - get_group_id(0) * get_local_size(0) + ty;
//     int out_j = get_group_id(1) * get_local_size(1) + tx;
//     uchar4 tttt = temp[ty][tx];
//     ((uchar3*)out_image)[out_i * rows + out_j] = make_uchar3(tttt.x, tttt.y, tttt.z);
// }

int main()
{
  int device_index = 0;
  cl_int errcode;

  Mat in_img = imread("pic.jpeg", CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! in_img.data )                              // Check for invalid input
  {
      cout << "Could not open or find the image" << std::endl ;
      return -1;
  }

  Mat out_img(in_img.cols, in_img.rows, DataType<Vec3b>::type);

  //код kernel-функции
  string sourceString = "\n\
  __kernel void turnmat(__global uchar* in_img, __global uchar* out_img, int rows, int cols)\n\
  {\n\
    int  i = get_global_id(1);\n\
    int  j = get_global_id(0);\n\
    if (i >= rows || j >= cols)\n\
      return;\n\
    int out_i = cols - 1 - j;\n\
    int out_j = i;\n\
    ((uchar3*)out_img)[out_i * rows + out_j] = ((uchar3*)in_img)[i * cols + j];\n\
  }";

  string shared_sourceString = "\n\
  __kernel void shared_turnmat(uchar *image, uchar *out_image, int rows, int cols)\n\
  {\n\
    __local uchar4 temp[shared_x][shared_y + 1];\n\
    int tx = get_local_id(0);\n\
    int ty = get_local_id(1);\n\
    int i = get_global_id(1);\n\
    int j = get_global_id(0);\n\
    if (i >= rows || j >= cols)\n\
        return;\n\
    int new_i = shared_x - 1 - tx;\n\
    int new_j = ty;\n\
    uchar3 ttt = ((uchar3*)image)[i * cols + j];\n\
    temp[new_i][new_j] = make_uchar4(ttt.x, ttt.y, ttt.z, 0);\n\
    \n\
    barrier();\n\
    \n\
    int out_i = cols - 1 - get_group_id(0) * get_local_size(0) + ty;\n\
    int out_j = get_group_id(1) * get_local_size(1) + tx;\n\
    uchar4 tttt = temp[ty][tx];\n\
    ((uchar3*)out_image)[out_i * rows + out_j] = make_uchar3(tttt.x, tttt.y, tttt.z);\n\
  }";

  //получаем список доступных OpenCL-платформ (драйверов OpenCL)
  std::vector<Platform> platform;//массив в который будут записываться идентификаторы платформ
  checkErrorEx( errcode = Platform::get(&platform) );
  cout << "OpenCL platforms found: " << platform.size() << "\n";
  cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>() << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << "\n";

  //в полученном списке платформ находим устройство GPU (видеокарту)
  std::vector<Device> devices;
  platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cout << "GPGPU devices found: " << devices.size() << "\n";
  if (devices.size() == 0)
  {
      cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead.\n";
      checkErrorEx( errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices) );
      cout << "CPU devices found: " << devices.size() << "\n";
      if (devices.size() == 0) {cout << "Error: CPU devices not found\n"; exit(-1);}
  }
  cout << "Use device N " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << "\n";

  //создаем контекст на видеокарте
  checkErrorEx( Context context(devices, NULL, NULL, NULL, &errcode) );

  //создаем очередь задач для контекста
  checkErrorEx( CommandQueue queue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode) );// третий параметр - свойства

  //создаем обьект-программу с заданным текстом программы
  checkErrorEx( Program program = Program(context, sourceString, false/*build*/, &errcode) );

  //компилируем и линкуем программу для видеокарты
  errcode = program.build(devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");
  if (errcode != CL_SUCCESS)
  {
      cout << "There were error during build kernel code. Please, check program code. Errcode = " << errcode << "\n";
      cout << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << "\n";
      return 1;
  }
  //создаем буфферы в видеопамяти
  checkErrorEx( Buffer dev_in_img = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 3 * in_img.rows * in_img.cols, in_img.data, &errcode ) );
  checkErrorEx( Buffer dev_out_img = Buffer( context, CL_MEM_READ_WRITE, 3 * in_img.rows * in_img.cols,  out_img.data, &errcode ) );

  //создаем объект - точку входа GPU-программы
  auto turnmat = KernelFunctor<Buffer, Buffer, int, int>(program, "turnmat");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  //EnqueueArgs enqueueArgs(queue, cl::NDRange(12*1024)/*globalSize*/, NullRange/*blockSize*/);
  int bx = 4, by = 32;
  EnqueueArgs enqueueArgs(queue, cl::NDRange((in_img.cols + (bx-1)) / bx, (in_img.rows + (by-1)) / by, 1)/*globalSize*/, NullRange/*cl::NDRange(bx, by, 1)blockSize*/);

  //запускаем и ждем
  clock_t t0 = clock();
  Event event = turnmat(enqueueArgs, dev_in_img, dev_out_img, in_img.rows, in_img.cols);
  checkErrorEx( errcode = event.wait() );
  clock_t t1 = clock();

  //считаем время
  cl_ulong time_start, time_end;
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start);
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end);
  double elapsedTimeGPU;
  if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE)
    elapsedTimeGPU = (double)(t1-t0)/CLOCKS_PER_SEC;
  else
  {
    checkError(event.getEventProfilingInfo);
    elapsedTimeGPU = (double)(time_end - time_start)/1e9;
  }

  cout << "GPU sum time = " << elapsedTimeGPU*1000 << " ms\n";
  cout << "GPU memory throughput = " << 6*in_img.cols*in_img.rows/elapsedTimeGPU/1024/1024/1024 << " Gb/s\n";

  imwrite("pic_resGPU.jpeg", out_img);
  return 0;
}
