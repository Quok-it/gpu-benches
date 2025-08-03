#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-error.h"
#include <iomanip>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <string>

using namespace std;

const int64_t max_buffer_size = 256l * 1024 * 1024 + 2;
double *dA, *dB, *dC, *dD;

struct BenchmarkResult {
  int block_size;
  int threads;
  double occupancy_percent;
  double init_gbps;
  double read_gbps;
  double scale_gbps;
  double triad_gbps;
  double pt3_gbps;
  double pt5_gbps;
};

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 const double *__restrict__ C,
                                 const double *__restrict__ D, const size_t N,
                                 bool secretlyFalse);

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.23;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void read_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  double temp = B[tidx];

  if (secretlyFalse || temp == 123.0)
    A[tidx] = temp + spoiler[tidx];
}

template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * 1.2;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void triad_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * D[tidx] + C[tidx];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void stencil1d3pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N - 1 || tidx == 0)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.5 * B[tidx - 1] - 1.0 * B[tidx] + 0.5 * B[tidx + 1];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}
template <typename T>
__global__ void stencil1d5pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N - 2 || tidx < 2)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.25 * B[tidx - 2] + 0.25 * B[tidx - 1] - 1.0 * B[tidx] +
            0.5 * B[tidx + 1] + 0.5 * B[tidx + 2];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}
double measureFunc(kernel_ptr_type func, int streamCount, int blockSize,
                   int blocksPerSM) {

#ifdef __NVCC__
  GPU_ERROR(cudaFuncSetAttribute(
      func, cudaFuncAttributePreferredSharedMemoryCarveout, 5));
#endif

  int maxActiveBlocks = 0;
  int spoilerSize = 1024;

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, func, 32, spoilerSize));

  while (maxActiveBlocks > blocksPerSM) {
    spoilerSize += 256;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, func, 32, spoilerSize));
    // std::cout << maxActiveBlocks << " " << spoilerSize << "\n";
  }

  /*if (maxActiveBlocks != blocksPerSM)
    std::cout << "Configure " << maxActiveBlocks << " instead of "
              << blocksPerSM << "\n";
*/

  MeasurementSeries time;

  func<<<max_buffer_size / blockSize + 1, blockSize, spoilerSize>>>(
      dA, dB, dC, dD, max_buffer_size, false);

  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  for (int iter = 0; iter < 31; iter++) {
    GPU_ERROR(cudaEventCreate(&start));
    GPU_ERROR(cudaEventCreate(&stop));
    GPU_ERROR(cudaEventRecord(start));
    func<<<max_buffer_size / blockSize + 1, blockSize, spoilerSize>>>(
        dA, dB, dC, dD, max_buffer_size, false);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    time.add(milliseconds / 1000);
  }

  double gbps = streamCount * max_buffer_size * sizeof(double) / time.median() * 1e-9;
  cout << fixed << setprecision(0) << setw(6) << " " << setw(5) << gbps;
  cout.flush();
  return gbps;
}

BenchmarkResult measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int blockSize,
                               int blocksPerSM) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  if (deviceName.starts_with("AMD Radeon RX 6")) {
    prop.maxThreadsPerMultiProcessor = 1024;
    prop.multiProcessorCount *= 2;
  }

  BenchmarkResult result = {};
  
  if (blockSize * blocksPerSM > prop.maxThreadsPerMultiProcessor ||
      blockSize > prop.maxThreadsPerBlock)
    return result;

  int smCount = prop.multiProcessorCount;
  int totalThreads = smCount * blockSize * blocksPerSM;
  double occupancy = (float)(blockSize * blocksPerSM) / prop.maxThreadsPerMultiProcessor * 100.0;
  
  // store result data
  result.block_size = blockSize;
  result.threads = totalThreads;
  result.occupancy_percent = occupancy;
  
  cout << setw(4) << blockSize << "   " << setw(7)
       << totalThreads << "  " << setw(5) << setw(6)
       << blocksPerSM << "  " << setprecision(1) << setw(5)
       << occupancy << "%     |  GB/s: ";

  // measure each kernel and store results
  result.init_gbps = measureFunc(kernels[0].first, kernels[0].second, blockSize, blocksPerSM);
  result.read_gbps = measureFunc(kernels[1].first, kernels[1].second, blockSize, blocksPerSM);
  result.scale_gbps = measureFunc(kernels[2].first, kernels[2].second, blockSize, blocksPerSM);
  result.triad_gbps = measureFunc(kernels[3].first, kernels[3].second, blockSize, blocksPerSM);
  result.pt3_gbps = measureFunc(kernels[4].first, kernels[4].second, blockSize, blocksPerSM);
  result.pt5_gbps = measureFunc(kernels[5].first, kernels[5].second, blockSize, blocksPerSM);

  cout << "\n";
  return result;
}

int main(int argc, char **argv) {
  int execution_id = 1; // default fallback
  string gpu_uuid = "unknown_gpu"; // default fallback
  
  // parse execution_id from command line
  if (argc > 1) {
    execution_id = atoi(argv[1]);
    if (execution_id <= 0) {
      cerr << "Error: Invalid execution_id provided: " << argv[1] << endl;
      return 1;
    }
  }
  
  // parse gpu_uuid from command line
  if (argc > 2) {
    gpu_uuid = string(argv[2]);
    if (gpu_uuid.empty()) {
      cerr << "Error: Empty gpu_uuid provided" << endl;
      return 1;
    }
  }
  
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dA, dA, dA, dA,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dB, dB, dB, dB,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dC, dC, dC, dC,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dD, dD, dD, dD,
                                                    max_buffer_size, false);
  GPU_ERROR(cudaDeviceSynchronize());

  vector<pair<kernel_ptr_type, int>> kernels = {
      {init_kernel<double>, 1},         {read_kernel<double>, 1},
      {scale_kernel<double>, 2},        {triad_kernel<double>, 4},
      {stencil1d3pt_kernel<double>, 2}, {stencil1d5pt_kernel<double>, 2}};

  vector<BenchmarkResult> results;

  cout << "block smBlocks   threads    occ%   |                init"
       << "       read       scale     triad       3pt        5pt\n";

  // Collect all results
  results.push_back(measureKernels(kernels, 16, 1));
  results.push_back(measureKernels(kernels, 32, 1));
  results.push_back(measureKernels(kernels, 48, 1));
  results.push_back(measureKernels(kernels, 64, 1));
  results.push_back(measureKernels(kernels, 80, 1));
  results.push_back(measureKernels(kernels, 96, 1));
  results.push_back(measureKernels(kernels, 112, 1));

  for (int warpCount = 4; warpCount <= 80; warpCount++) {
    int threadCount = warpCount * 32;
    if (threadCount / 32 % 2 == 0) {
      // and (warpCount < 16 || warpCount % 8 == 0))
      auto result = measureKernels(kernels, threadCount / 2, 2);
      if (result.block_size > 0) results.push_back(result);
    } else if (warpCount < 6) {
      auto result = measureKernels(kernels, threadCount, 1);
      if (result.block_size > 0) results.push_back(result);
    }
  }

  // output single SQL INSERT statement with all results
  cout << "\n--- SQL INSERT Statement ---\n";
  cout << "-- GPU Stream Benchmark Results (All Configs)\n";
  cout << "-- Generated at: " << time(nullptr) << "\n\n";
  
  // filter valid results
  vector<BenchmarkResult> validResults;
  for (const auto& result : results) {
    if (result.block_size > 0) {
      validResults.push_back(result);
    }
  }
  
  if (!validResults.empty()) {
    cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
    cout << "  " << execution_id << ", -- execution_id\n";
    cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
    cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'gpu_stream'),\n";
    cout << "  NOW(),\n";
    cout << "  'memory',\n";
    cout << "  'gpu_stream_bandwidth',\n";
    cout << "  '{\n";
    cout << "    \"configurations\": [\n";
    
    for (size_t i = 0; i < validResults.size(); ++i) {
      const auto& result = validResults[i];
      cout << "      {\n";
      cout << "        \"block_size\": " << result.block_size << ",\n";
      cout << "        \"threads\": " << result.threads << ",\n";
      cout << "        \"occupancy_percent\": " << fixed << setprecision(1) << result.occupancy_percent << ",\n";
      cout << "        \"init_gbps\": " << fixed << setprecision(0) << result.init_gbps << ",\n";
      cout << "        \"read_gbps\": " << result.read_gbps << ",\n";
      cout << "        \"scale_gbps\": " << result.scale_gbps << ",\n";
      cout << "        \"triad_gbps\": " << result.triad_gbps << ",\n";
      cout << "        \"3pt_gbps\": " << result.pt3_gbps << ",\n";
      cout << "        \"5pt_gbps\": " << result.pt5_gbps << "\n";
      cout << "      }";
      if (i < validResults.size() - 1) cout << ",";
      cout << "\n";
    }
    
    cout << "    ],\n";
    cout << "    \"total_configurations\": " << validResults.size() << ",\n";
    cout << "    \"max_bandwidth\": {\n";
    
    // Find maximum values across all configurations
    double max_init = 0, max_read = 0, max_scale = 0, max_triad = 0, max_3pt = 0, max_5pt = 0;
    for (const auto& result : validResults) {
      max_init = max(max_init, result.init_gbps);
      max_read = max(max_read, result.read_gbps);
      max_scale = max(max_scale, result.scale_gbps);
      max_triad = max(max_triad, result.triad_gbps);
      max_3pt = max(max_3pt, result.pt3_gbps);
      max_5pt = max(max_5pt, result.pt5_gbps);
    }
    
    cout << "      \"init_gbps\": " << fixed << setprecision(0) << max_init << ",\n";
    cout << "      \"read_gbps\": " << max_read << ",\n";
    cout << "      \"scale_gbps\": " << max_scale << ",\n";
    cout << "      \"triad_gbps\": " << max_triad << ",\n";
    cout << "      \"3pt_gbps\": " << max_3pt << ",\n";
    cout << "      \"5pt_gbps\": " << max_5pt << "\n";
    cout << "    }\n";
    cout << "  }'::jsonb\n";
    cout << ");\n\n";
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
