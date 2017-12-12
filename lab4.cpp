#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <iostream>
#include <time.h>
#include <string.h>

using namespace cl;
using namespace std;

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    cout << "Error in " #func "\nError code = " << errcode << "\n";\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);

// __kernel void turnmat(cl_uchar *image, cl_uchar *out_image, cl_int rows, cl_int cols)
// {
//     int i = get_global_id(1);
//     int j = get_global_id(0);
//     if (i >= rows || j >= cols)
//         return;
//
//     cl_int out_i = cols - 1 - j;
//     cl_int out_j = i;
//
//     ((cl_uchar3*)out_image)[out_i * rows + out_j] = ((cl_uchar3*)image)[i * cols + j];
// }

#define shared_x 16
#define shared_y 16

// __kernel void shared_turnmat(cl_uchar *image, cl_uchar *out_image, cl_int rows, cl_int cols)
// {
//     __local cl_uchar4 temp[shared_x][shared_y + 1];
//     cl_int tx = get_local_id(0);
//     cl_int ty = get_local_id(1);
//     int i = get_global_id(1);
//     int j = get_global_id(0);
//     if (i >= rows || j >= cols)
//         return;
//
//     cl_int new_i = shared_x - 1 - tx;
//     cl_int new_j = ty;
//     cl_uchar3 ttt = ((cl_uchar3*)image)[i * cols + j];
//     temp[new_i][new_j] = make_uchar4(ttt.x, ttt.y, ttt.z, 0);
//
//      barrier();
//
//     cl_int out_i = cols - 1 - get_group_id(0) * get_local_size(0) + ty;
//     int out_j = get_group_id(1) * get_local_size(1) + tx;
//     cl_uchar4 tttt = temp[ty][tx];
//     ((cl_uchar3*)out_image)[out_i * rows + out_j] = make_uchar3(tttt.x, tttt.y, tttt.z); // ???
// }

int main()
{
  //код kernel-функции
  string sourceString = "\n\
  __kernel void sum(__global float *a, __global float *b, __global float *c, int N)\n\
  {\n\
    int  id = get_global_id(0);\n\
    int threadsNum = get_global_size(0);\n\
    for (int i = id; i < N; i += threadsNum)\n\
      c[i] = a[i]+b[i];\n\
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
  checkErrorEx( Buffer dev_a = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N*sizeof(float),  host_a, &errcode ) );
  checkErrorEx( Buffer dev_b = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N*sizeof(float),  host_b, &errcode ) );
  checkErrorEx( Buffer dev_c = Buffer( context, CL_MEM_READ_WRITE, N*sizeof(float),  NULL, &errcode ) );

  //создаем объект - точку входа GPU-программы
  auto sum = KernelFunctor<Buffer, Buffer, Buffer, int>(program, "sum");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  EnqueueArgs enqueueArgs(queue, cl::NDRange(12*1024)/*globalSize*/, NullRange/*blockSize*/);

  //запускаем и ждем
  clock_t t0 = clock();
  Event event = sum(enqueueArgs, dev_a, dev_b, dev_c, N);
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
  checkErrorEx( errcode = queue.enqueueReadBuffer(dev_c, true, 0, N*sizeof(float), host_c, NULL, NULL) );
  // check
  for (int i = 0; i < N; i++)
      if (::abs(host_c[i] - host_c_check[i]) > 1e-6)
      {
          cout << "Error in element N " << i << ": c[i] = " << host_c[i] << " c_check[i] = " << host_c_check[i] << "\n";
          exit(1);
      }
  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
  cout << "CPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";
  cout << "GPU sum time = " << elapsedTimeGPU*1000 << " ms\n";
  cout << "GPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeGPU/1024/1024/1024 << " Gb/s\n";
  return 0;
}
