#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-error.h"
#include <cooperative_groups.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

namespace cg = cooperative_groups;

using namespace std;

// global variables for SQL output mode
string execution_id = "1";
string gpu_uuid = "unknown_gpu";
bool sql_output_mode = false;

struct SmallKernelResult {
  int problem_size;
  double memory_size_kb;
  double block_32_gbps;
  double block_64_gbps;
  double block_128_gbps;
  double block_256_gbps;
  double block_512_gbps;
  double block_1024_gbps;
};

// global results storage
vector<SmallKernelResult> kernel_results;

const int64_t max_buffer_count = 512l * 1024 * 1024 + 2;
double *dA, *dB;

#ifdef __NVCC__
const int spoilerSize = 768;
#else
const int spoilerSize = 4 * 1024;
#endif

bool useCudaGraph = false;
bool usePersistentThreadsAtomic = false;
bool usePersistentThreadsGsync = false;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 int size);

template <typename T> __global__ void init_kernel(T *A, size_t N, T val) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (T)val;
  }
}

template <typename T>
__global__ void scale(T *__restrict__ A, const T *__restrict__ B,
                      const int size) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= size)
    return;

  A[tidx] = B[tidx] * 0.25;
}

template <typename T>
__global__ void sync_kernel_gsync(T *__restrict__ A, T *__restrict__ B,
                                  int size, int iters) {

  cg::grid_group g = cg::this_grid();
  int tidx = g.thread_rank();

  for (int iter = 0; iter < iters; iter++) {
    for (int id = tidx; id < size; id += blockDim.x * gridDim.x) {
      A[id] = B[id] * 0.25;
    }

    g.sync();
  }
}
template <typename T>
__global__ void sync_kernel_atomic(volatile int *flags, T *__restrict__ A,
                                   T *__restrict__ B, int size, int iters) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int threadCount = gridDim.x;

  for (int iter = 0; iter < iters; iter++) {
    for (int id = tidx; id < size; id += blockDim.x * gridDim.x) {

      A[id] = B[id] * 0.25;
    }

    __syncthreads();
    __threadfence();
    int old_val;
    if (threadIdx.x == 0) {
      old_val = atomicAdd((int *)&(flags[iter]), 1);
      while (flags[iter] != threadCount)
        ;
    }
    __syncthreads();
  }
}

double measureFunc(kernel_ptr_type func, int size, dim3 blockSize) {
  MeasurementSeries time;

  dim3 grid = size / blockSize.x + 1;

  func<<<grid, blockSize>>>(dA, dB, size);

  int iters = min(30000, max(2000, 100000 * 10000 / size));

  if (usePersistentThreadsGsync) {

    cudaDeviceProp prop;
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    std::string deviceName = prop.name;
    int smCount = prop.multiProcessorCount;
    int maxActiveBlocks = 0;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, sync_kernel_gsync<double>,
        blockSize.x * blockSize.y * blockSize.z, 0));

    const int blockCount = min(size / (blockSize.x * blockSize.y * blockSize.z),
                               smCount * maxActiveBlocks);

    for (int iter = 0; iter < 3; iter++) {

      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();

      void *kernelArgs[] = {&dA, &dB, &size, &iters};
      GPU_ERROR(cudaLaunchCooperativeKernel((void *)sync_kernel_gsync<double>,
                                            blockCount, blockSize, kernelArgs,
                                            0, 0));

      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
  } else if (usePersistentThreadsAtomic) {

    int *flags;

    GPU_ERROR(cudaMalloc(&flags, sizeof(int) * iters));

    cudaDeviceProp prop;
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    std::string deviceName = prop.name;
    int smCount = prop.multiProcessorCount;
    int maxActiveBlocks = 0;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, sync_kernel_atomic<double>,
        blockSize.x * blockSize.y * blockSize.z, 0));

    const int blockCount = min(size / (blockSize.x * blockSize.y * blockSize.z),
                               smCount * maxActiveBlocks);

    for (int iter = 0; iter < 3; iter++) {
      init_kernel<<<52, 256>>>(flags, iters, 0);

      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();

      sync_kernel_atomic<double>
          <<<blockCount, blockSize>>>(flags, dA, dB, size, iters);

      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
    GPU_ERROR(cudaFree(flags));

  } else if (useCudaGraph) {

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStream_t stream;
    GPU_ERROR(cudaStreamCreate(&stream));

    GPU_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < iters; i += 2) {
      func<<<grid, blockSize, 0, stream>>>(dA, dB, size);
      func<<<grid, blockSize, 0, stream>>>(dB, dA, size);
    }
    GPU_ERROR(cudaStreamEndCapture(stream, &graph));
    GPU_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    for (int iter = 0; iter < 3; iter++) {
      GPU_ERROR(cudaStreamSynchronize(stream));
      double t1 = dtime();
      GPU_ERROR(cudaGraphLaunch(instance, stream));
      GPU_ERROR(cudaStreamSynchronize(stream));
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
    GPU_ERROR(cudaStreamDestroy(stream));
    GPU_ERROR(cudaGraphDestroy(graph));
    GPU_ERROR(cudaGraphExecDestroy(instance));
  } else {

    func<<<grid, blockSize, 0>>>(dA, dB, size);
    for (int iter = 0; iter < 3; iter++) {
      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();
      for (int i = 0; i < iters; i += 2) {
        func<<<grid, blockSize, 0>>>(dA, dB, size);
        func<<<grid, blockSize, 0>>>(dB, dA, size);
      }
      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
  }
  
  double bandwidth_gbps = size * 2 * sizeof(double) / time.median() * 1e-9;
  
  if (!sql_output_mode) {
    cout << fixed << setprecision(0) << setw(6) << setw(5) << bandwidth_gbps << "  ";
    cout.flush();
  }
  
  return bandwidth_gbps;
}

int main(int argc, char **argv) {
  // check for execution_id and gpu_uuid (SQL mode) first
  bool found_sql_args = false;
  for (int i = 1; i < argc; i++) {
    // if find non-empty argument --> execution_id
    string potential_execution_id = string(argv[i]);
    if (!potential_execution_id.empty()) {
      execution_id = potential_execution_id;
      sql_output_mode = true;
      found_sql_args = true;
      
      // check for gpu_uuid as next argument
      if (i + 1 < argc) {
        gpu_uuid = string(argv[i + 1]);
        if (gpu_uuid.empty()) {
          cerr << "Error: Empty gpu_uuid provided" << endl;
          return 1;
        }
      }
      break;
    }
  }
  
  // handle benchmark mode flags only if not in SQL mode
  if (!found_sql_args) {
    if (argc > 1 && string(argv[1]) == "-graph") {
      useCudaGraph = true;
    }
    if (argc > 1 && string(argv[1]) == "-pta") {
      usePersistentThreadsAtomic = true;
    }
    if (argc > 1 && string(argv[1]) == "-pt-gsync") {
      usePersistentThreadsGsync = true;
    }
  }

  GPU_ERROR(cudaMalloc(&dA, max_buffer_count * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_count * sizeof(double)));

  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dA, max_buffer_count,
                                                     0.23);
  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dB, max_buffer_count,
                                                     1.44);

  GPU_ERROR(cudaDeviceSynchronize());

  for (int d = 4 * 1024; d < 8 * 16 * 1024 * 1024;
       d += std::max((int)1, (int)(d * 0.06))) {

    SmallKernelResult result;
    result.problem_size = d;
    result.memory_size_kb = d * sizeof(double) * 2 / 1024.0;

    if (!sql_output_mode) {
      std::cout << d << "  " << d * sizeof(double) * 2 / 1024 << "kB  ";
    }

    // measure performance for each block size
    result.block_32_gbps = measureFunc(scale<double>, d, dim3(32, 1, 1));
    result.block_64_gbps = measureFunc(scale<double>, d, dim3(64, 1, 1));
    result.block_128_gbps = measureFunc(scale<double>, d, dim3(128, 1, 1));
    result.block_256_gbps = measureFunc(scale<double>, d, dim3(256, 1, 1));
    result.block_512_gbps = measureFunc(scale<double>, d, dim3(512, 1, 1));
    result.block_1024_gbps = measureFunc(scale<double>, d, dim3(1024, 1, 1));

    kernel_results.push_back(result);

    if (!sql_output_mode) {
      std::cout << "\n";
      std::cout.flush();
    }
  }

  if (sql_output_mode) {
    // gen SQL INSERT statement
    cout << "-- GPU Small Kernels Benchmark Results\n";
    cout << "-- Generated at: " << time(nullptr) << "\n\n";
    
    if (!kernel_results.empty()) {
      // calc summary stats
      double max_bandwidth = 0;
      int optimal_block_size = 32;
      int optimal_problem_size = 0;
      
      // find peak performance for each block size and overall
      double block_peaks[6] = {0, 0, 0, 0, 0, 0}; // 32, 64, 128, 256, 512, 1024
      
      for (const auto& result : kernel_results) {
        double bandwidths[6] = {
          result.block_32_gbps, result.block_64_gbps, result.block_128_gbps,
          result.block_256_gbps, result.block_512_gbps, result.block_1024_gbps
        };
        
        for (int i = 0; i < 6; i++) {
          if (bandwidths[i] > block_peaks[i]) {
            block_peaks[i] = bandwidths[i];
          }
          if (bandwidths[i] > max_bandwidth) {
            max_bandwidth = bandwidths[i];
            optimal_block_size = 32 << i; // 32 * 2^i
            optimal_problem_size = result.problem_size;
          }
        }
      }
      
      cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
      cout << "  '" << execution_id << "', -- execution_id\n";
      cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
      cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'gpu_small_kernels'),\n";
      cout << "  NOW(),\n";
      cout << "  'system',\n";
      cout << "  'small_kernel_performance',\n";
      cout << "  '{\n";
      cout << "    \"configurations\": [\n";
      
      for (size_t i = 0; i < kernel_results.size(); ++i) {
        const auto& result = kernel_results[i];
        cout << "      {\n";
        cout << "        \"problem_size\": " << result.problem_size << ",\n";
        cout << "        \"memory_size_kb\": " << fixed << setprecision(1) << result.memory_size_kb << ",\n";
        cout << "        \"block_performances\": {\n";
        cout << "          \"block_32\": " << fixed << setprecision(0) << result.block_32_gbps << ",\n";
        cout << "          \"block_64\": " << result.block_64_gbps << ",\n";
        cout << "          \"block_128\": " << result.block_128_gbps << ",\n";
        cout << "          \"block_256\": " << result.block_256_gbps << ",\n";
        cout << "          \"block_512\": " << result.block_512_gbps << ",\n";
        cout << "          \"block_1024\": " << result.block_1024_gbps << "\n";
        cout << "        }\n";
        cout << "      }";
        if (i < kernel_results.size() - 1) cout << ",";
        cout << "\n";
      }
      
      cout << "    ],\n";
      cout << "    \"summary\": {\n";
      cout << "      \"total_configurations\": " << kernel_results.size() << ",\n";
      cout << "      \"size_range\": {\n";
      cout << "        \"min_problem_size\": " << kernel_results.front().problem_size << ",\n";
      cout << "        \"max_problem_size\": " << kernel_results.back().problem_size << ",\n";
      cout << "        \"min_memory_kb\": " << fixed << setprecision(1) << kernel_results.front().memory_size_kb << ",\n";
      cout << "        \"max_memory_kb\": " << kernel_results.back().memory_size_kb << "\n";
      cout << "      },\n";
      cout << "      \"peak_performance\": {\n";
      cout << "        \"max_bandwidth_gbps\": " << fixed << setprecision(0) << max_bandwidth << ",\n";
      cout << "        \"optimal_block_size\": " << optimal_block_size << ",\n";
      cout << "        \"optimal_problem_size\": " << optimal_problem_size << ",\n";
      cout << "        \"performance_by_block_size\": {\n";
      cout << "          \"block_32_peak\": " << block_peaks[0] << ",\n";
      cout << "          \"block_64_peak\": " << block_peaks[1] << ",\n";
      cout << "          \"block_128_peak\": " << block_peaks[2] << ",\n";
      cout << "          \"block_256_peak\": " << block_peaks[3] << ",\n";
      cout << "          \"block_512_peak\": " << block_peaks[4] << ",\n";
      cout << "          \"block_1024_peak\": " << block_peaks[5] << "\n";
      cout << "        }\n";
      cout << "      }\n";
      cout << "    }\n";
      cout << "  }'::jsonb\n";
      cout << ");\n\n";
    }
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
