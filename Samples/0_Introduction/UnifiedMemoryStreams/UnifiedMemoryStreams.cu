/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample implements a simple task consumer using threads and streams
 * with all data in Unified Memory, and tasks consumed by both host and device
 */

/*
* 此示例使用线程和流实现了一个简单的任务消费者
* 所有数据都在统一内存中，任务由主机和设备共同消耗
*/

/*
* Unified Memory介绍
* cuda unified memory是一种在CUDA编程模型中使用的内存管理机制。
* 它允许开发者以一种统一的方式访问和管理CPU和GPU的内存，从而简化了编程的复杂性。
* 在Unified Memory模型中，CUDA会自动管理数据在CPU和GPU之间的迁移，使得数据能够在需要的时候出现在正确的位置上。
* 这种机制消除了手动管理数据传输的需要，使得代码更加简洁和易于维护。
* 此外，Unified Memory还提供了内存超额订阅功能，即可以分配超过物理GPU内存大小的内存空间，CUDA会在需要时自动进行数据的换入换出。
*/

// system includes
#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#ifdef USE_PTHREADS
#include <pthread.h>
#else
#include <omp.h>
#endif
#include <stdlib.h>

// cuBLAS
#include <cublas_v2.h>

// utilities
#include <helper_cuda.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent
// functions
void srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }
#endif

const char *sSDKname = "UnifiedMemoryStreams";

// simple task
template <typename T>
struct Task {
  unsigned int size, id;
  T *data;
  T *result;
  T *vector;

  Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL){};
  Task(unsigned int s) : size(s), id(0), data(NULL), result(NULL) {
    // allocate unified memory -- the operation performed in this example will
    // be a DGEMV
    // 这个例子中的逻辑是执行DGEMV
    // DGEMV 密集型通用矩阵-向量乘法（Dense General Matrix-Vector Multiplication）

    /* 
    * cudaMallocManaged函数介绍
    * 是 CUDA 编程中的一个重要函数，它用于在 CUDA 程序中分配一块可由 CPU 和 GPU 共同访问和管理的内存。
    * 这种内存称为托管内存（managed memory）。
    * 使用托管内存可以大大简化数据在 CPU 和 GPU 之间的传输过程，因为 CUDA 运行时会自动处理这些内存的数据迁移，从而减少了程序员的负担。
    */
    checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
    checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));

    /*
    * cudaDeviceSynchronize函数介绍
    * 是CUDA Runtime API中的一个函数，用于等待直到所有之前提交的CUDA命令都完成。
    * 这包括在调用cudaDeviceSynchronize之前的所有CUDA内核启动、内存复制和内存分配操作等。
    */
    checkCudaErrors(cudaDeviceSynchronize());
  }

  ~Task() {
    // ensure all memory is deallocated
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaFree(vector));
  }

  // 若是通过无参构造函数Task()构造的Task，可以继续调用allocate来分配unified memory
  void allocate(const unsigned int s, const unsigned int unique_id) {
    // allocate unified memory outside of constructor
    id = unique_id;
    size = s;
    checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
    checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
    checkCudaErrors(cudaDeviceSynchronize());

    // populate data with random elements
    // 数据初始化，随机值
    for (unsigned int i = 0; i < size * size; i++) {
      data[i] = drand48();
    }

    for (unsigned int i = 0; i < size; i++) {
      result[i] = 0.;
      vector[i] = drand48();
    }
  }
};

#ifdef USE_PTHREADS
struct threadData_t {
  int tid;
  Task<double> *TaskListPtr;
  cudaStream_t *streams;
  cublasHandle_t *handles;
  int taskSize;
};

typedef struct threadData_t threadData;
#endif

// simple host dgemv: assume data is in row-major format and square
template <typename T>
// 在CPU上执行dgemv，假设数据是行优先存储，且矩阵是个方阵
/*
* [row-major format]介绍
* 概述：将多维数组打平成一维数组
* 在计算机科学中，多维数组（如矩阵）可以以行优先或列优先的方式存储在内存中。
* 在行优先格式中，数组的元素首先按行填充，即首先填充第一行的所有元素，然后是第二行，依此类推。（此处举例是二维矩阵，按行打平）
*/
void gemv(int m, int n, T alpha, T *A, T *x, T beta, T *result) {
  // rows
  for (int i = 0; i < n; i++) {
    result[i] *= beta;

    for (int j = 0; j < n; j++) {
      result[i] += A[i * n + j] * x[j];
    }
  }
}

// execute a single task on either host or device depending on size
// 执行一个task，根据task的size决定是在CPU还是在GPU上运行
// size < 100 在CPU上执行，否则在GPU上执行
#ifdef USE_PTHREADS
void *execute(void *inpArgs) {
  threadData *dataPtr = (threadData *)inpArgs;
  cudaStream_t *stream = dataPtr->streams;
  cublasHandle_t *handle = dataPtr->handles;
  int tid = dataPtr->tid;

  for (int i = 0; i < dataPtr->taskSize; i++) {
    Task<double> &t = dataPtr->TaskListPtr[i];

    if (t.size < 100) {
      // perform on host
      printf("Task [%d], thread [%d] executing on host (%d)\n", t.id, tid,
             t.size);

      // attach managed memory to a (dummy) stream to allow host access while
      // the device is running
      checkCudaErrors(
          cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
      checkCudaErrors(
          cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
      checkCudaErrors(
          cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
      // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
      checkCudaErrors(cudaStreamSynchronize(stream[0]));
      // call the host operation
      gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    } else {
      // perform on device
      printf("Task [%d], thread [%d] executing on device (%d)\n", t.id, tid,
             t.size);
      double one = 1.0;
      double zero = 0.0;

      // attach managed memory to my stream
      checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0,
                                               cudaMemAttachSingle));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0,
                                               cudaMemAttachSingle));
      checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0,
                                               cudaMemAttachSingle));
      // call the device operation
      checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size,
                                  &one, t.data, t.size, t.vector, 1, &zero,
                                  t.result, 1));
    }
  }

  pthread_exit(NULL);
}
#else
template <typename T>
void execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream,
             int tid) {
  if (t.size < 100) {
    // perform on host
    printf("Task [%d], thread [%d] executing on host (%d)\n", t.id, tid,
           t.size);

    // attach managed memory to a (dummy) stream to allow host access while the
    // device is running
    checkCudaErrors(
        cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
    checkCudaErrors(
        cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
    checkCudaErrors(
        cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
    // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
    checkCudaErrors(cudaStreamSynchronize(stream[0]));
    // call the host operation
    gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
  } else {
    // perform on device
    printf("Task [%d], thread [%d] executing on device (%d)\n", t.id, tid,
           t.size);
    double one = 1.0;
    double zero = 0.0;

    // attach managed memory to my stream
    checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0,
                                             cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0,
                                             cudaMemAttachSingle));
    checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0,
                                             cudaMemAttachSingle));
    // call the device operation
    checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size,
                                &one, t.data, t.size, t.vector, 1, &zero,
                                t.result, 1));
  }
}
#endif

// populate a list of tasks with random sizes
// 初始化Task，其size被初始化为随机数
template <typename T>
void initialise_tasks(std::vector<Task<T> > &TaskList) {
  for (unsigned int i = 0; i < TaskList.size(); i++) {
    // generate random size
    int size;
    size = std::max((int)(drand48() * 1000.0), 64);
    TaskList[i].allocate(size, i);
  }
}

int main(int argc, char **argv) {
  // set device
  cudaDeviceProp device_prop;
  int dev_id = findCudaDevice(argc, (const char **)argv);
  checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

  if (!device_prop.managedMemory) {
    // This samples requires being run on a device that supports Unified Memory
    fprintf(stderr, "Unified Memory not supported on this device\n");

    exit(EXIT_WAIVED);
  }

  if (device_prop.computeMode == cudaComputeModeProhibited) {
    // This sample requires being run with a default or process exclusive mode
    fprintf(stderr,
            "This sample requires a device in either default or process "
            "exclusive mode\n");

    exit(EXIT_WAIVED);
  }

  // randomise task sizes
  int seed = (int)time(NULL);
  srand48(seed);

  // set number of threads
  const int nthreads = 4;

  // number of streams = number of threads
  /*
  * [stream]介绍
  * CUDA的stream（流）是一种处理并行操作的方式，它允许开发者在CUDA程序中组织和管理并发执行的任务。
  * 通过使用不同的流，可以并行执行多个核函数（kernels）或内存传输操作，从而实现更高的硬件利用率和程序性能。
  * 此处创建了streams数组，可以让多个thread在不同的stream中并发执行（并发的内存操作，并发的kernel执行）
  */
  cudaStream_t *streams = new cudaStream_t[nthreads + 1];
  cublasHandle_t *handles = new cublasHandle_t[nthreads + 1];

  for (int i = 0; i < nthreads + 1; i++) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
    checkCudaErrors(cublasCreate(&handles[i]));
  }

  // create list of N tasks
  unsigned int N = 40;
  std::vector<Task<double> > TaskList(N);
  initialise_tasks(TaskList);

  printf("Executing tasks on host / device\n");

// run through all tasks using threads and streams
#ifdef USE_PTHREADS
  pthread_t threads[nthreads];
  threadData *InputToThreads = new threadData[nthreads];

  for (int i = 0; i < nthreads; i++) {
    checkCudaErrors(cudaSetDevice(dev_id));
    InputToThreads[i].tid = i;
    InputToThreads[i].streams = streams;
    InputToThreads[i].handles = handles;

    if ((TaskList.size() / nthreads) == 0) {
      InputToThreads[i].taskSize = (TaskList.size() / nthreads);
      InputToThreads[i].TaskListPtr =
          &TaskList[i * (TaskList.size() / nthreads)];
    } else {
      if (i == nthreads - 1) {
        InputToThreads[i].taskSize =
            (TaskList.size() / nthreads) + (TaskList.size() % nthreads);
        InputToThreads[i].TaskListPtr =
            &TaskList[i * (TaskList.size() / nthreads) +
                      (TaskList.size() % nthreads)];
      } else {
        InputToThreads[i].taskSize = (TaskList.size() / nthreads);
        InputToThreads[i].TaskListPtr =
            &TaskList[i * (TaskList.size() / nthreads)];
      }
    }

    pthread_create(&threads[i], NULL, &execute, &InputToThreads[i]);
  }
  for (int i = 0; i < nthreads; i++) {
    pthread_join(threads[i], NULL);
  }
#else
  omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < TaskList.size(); i++) {
    checkCudaErrors(cudaSetDevice(dev_id));
    int tid = omp_get_thread_num();
    execute(TaskList[i], handles, streams, tid);
  }
#endif

  cudaDeviceSynchronize();

  // Destroy CUDA Streams, cuBlas handles
  for (int i = 0; i < nthreads + 1; i++) {
    cudaStreamDestroy(streams[i]);
    cublasDestroy(handles[i]);
  }

  // Free TaskList
  std::vector<Task<double> >().swap(TaskList);

  printf("All Done!\n");
  exit(EXIT_SUCCESS);
}
