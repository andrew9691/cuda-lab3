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

#define shared_x 16
#define shared_y 16

struct pix3 {char r, g, b;};

int main()
{
  cout << "size = " << sizeof(pix3) << endl;
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
  struct pix{ char r, g, b; };\n\
  __kernel void turnmat(__global uchar* in_img, __global uchar* out_img, int rows, int cols)\n\
  {\n\
    int  i = get_global_id(1);\n\
    int  j = get_global_id(0);\n\
    if (i >= rows || j >= cols)\n\
      return;\n\
    int out_i = cols - 1 - j;\n\
    int out_j = i;\n\
    ((__global struct pix*)out_img)[out_i * rows + out_j] = ((__global struct pix*)in_img)[i * cols + j];\n\
  }";

  string shared_sourceString = "\n\
  #define shared_x 16\n\
  #define shared_y 16\n\
  struct pix3{ char r, g, b; };\n\
  __kernel void shared_turnmat(__global uchar *image, __global uchar *out_image, int rows, int cols, __global int* count)\n\
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
    struct pix3 ttt = ((__global struct pix3*)image)[i * cols + j];\n\
    temp[new_i][new_j] = (uchar4)(ttt.r, ttt.g, ttt.b, 0);\n\
    \n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    \n\
    int out_i = cols - 1 - (1 + get_group_id(0)) * get_local_size(0) + ty;\n\
    int out_j = get_group_id(1) * get_local_size(1) + tx;\n\
    uchar4 tttt = temp[ty][tx];\n\
    struct pix3 t = {tttt.x, tttt.y, tttt.z};\n\
    int ind = out_i * rows + out_j;\n\
    //if (ind >= 0 && ind < rows*cols)\n\
      ((__global struct pix3*)out_image)[ind] = t;\n\
    //if (ind >= rows*cols)\n\
      //*count += 1;\n\
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
  //checkErrorEx( Program program = Program(context, sourceString, false/*build*/, &errcode) ); //////////////////////////////////////////////////////////////////////////////
  checkErrorEx( Program program = Program(context, shared_sourceString, false/*build*/, &errcode) );

  //компилируем и линкуем программу для видеокарты
  errcode = program.build(devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");
  if (errcode != CL_SUCCESS)
  {
      cout << "There were error during build kernel code. Please, check program code. Errcode = " << errcode << "\n";
      cout << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << "\n";
      return 1;
  }

  int ctn = 0;
  //создаем буфферы в видеопамяти
  checkErrorEx( Buffer dev_in_img = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (size_t)3 * in_img.rows * in_img.cols, in_img.data, &errcode ) );
  checkErrorEx( Buffer dev_out_img = Buffer( context, CL_MEM_READ_WRITE, (size_t)3 * in_img.rows * in_img.cols, out_img.data, &errcode ) );
  checkErrorEx( Buffer count = Buffer( context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, (size_t)4, &ctn, &errcode ) );

  //создаем объект - точку входа GPU-программы
  //auto turnmat = KernelFunctor<Buffer, Buffer, int, int>(program, "turnmat"); ////////////////////////////////////////////////////////////////////////////////////////////////////
  auto shared_turnmat = KernelFunctor<Buffer, Buffer, int, int, Buffer>(program, "shared_turnmat");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  //EnqueueArgs enqueueArgs(queue, cl::NDRange(in_img.cols, in_img.rows), NullRange); /////////////////////////////////////
  EnqueueArgs enqueueArgs(queue, cl::NDRange(in_img.cols, in_img.rows), cl::NDRange(shared_x, shared_y));

  //запускаем и ждем
  clock_t t0 = clock();

  //Event event = turnmat(enqueueArgs, dev_in_img, dev_out_img, in_img.rows, in_img.cols); ////////////////////////////////////////////////////////////////////////////////////////////////////
  Event event = shared_turnmat(enqueueArgs, dev_in_img, dev_out_img, in_img.rows, in_img.cols, count);
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

  checkErrorEx( errcode = queue.enqueueReadBuffer(dev_out_img, true, 0, (size_t)3 * in_img.rows * in_img.cols, out_img.data, NULL, NULL) );
  checkErrorEx( errcode = queue.enqueueReadBuffer(count, true, 0, (size_t)4, &ctn, NULL, NULL) );
  cout << "count = " << ctn << endl;
  imwrite("pic_resGPU.jpeg", out_img);
  return 0;
}
