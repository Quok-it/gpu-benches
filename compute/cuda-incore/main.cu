#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-clock.cuh"
#include "../../gpu-error.h"
#include "../../metrics.cuh"
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <ctime>

using namespace std;

// global variables for SQL output mode
int execution_id = 1;
string gpu_uuid = "unknown_gpu";
bool sql_output_mode = false;

struct InCoreResult {
  string operation_name;
  string precision;
  int warp_count;
  int stream_count;
  double reciprocal_throughput;
};

// global results storage
vector<InCoreResult> incore_results;

template <typename T> __global__ void initKernel(T *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <typename T, int N, int M>
__global__ void FMA_mixed(T p, T *A, int iters) {
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    T t[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
      t[m] = p + threadIdx.x + iter + m;
    }
#pragma unroll
    for (int n = 0; n < N / M; n++) {
#pragma unroll
      for (int m = 0; m < M; m++) {
        t[m] = t[m] * (T)0.9 + (T)0.5;
      }
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (t[m] > (T)22313.0) {
        A[0] = t[m];
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void FMA_separated(T p, T *A, int iters) {

  for (int iter = 0; iter < iters; iter++) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;
      for (int n = 0; n < N; n++) {
        t = t * (T)0.9 + (T)0.5;
      }
      if (t > (T)22313.0) {
        A[0] = t;
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void DIV_separated(T p, T *A, int iters) {

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;

      for (int n = 0; n < N; n++) {
        t = 0.1 / (t + 0.2);
      }

      A[threadIdx.x + iter] = t;
    }
  }
}

template <typename T, int N, int M>
__global__ void SQRT_separated(T p, T *A, int iters) {

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {

    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;

      for (int n = 0; n < N; n++) {
        t = sqrt(t + 0.2);
      }

      A[threadIdx.x + iter] = t;
    }
  }
}

unsigned int gpu_clock = 0;

template <typename T, int N, int M>
double measure(int warpCount, void (*kernel)(T, T *, int)) {
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(0, &device);

  const int iters = 10000;
  const int blockSize = 32 * warpCount;
  const int blockCount = 1;

  MeasurementSeries time;

  T *dA;
  GPU_ERROR(cudaMalloc(&dA, iters * 2 * sizeof(T)));
  initKernel<<<52, 256>>>(dA, iters * 2);
  GPU_ERROR(cudaDeviceSynchronize());

  kernel<<<blockCount, blockSize>>>((T)0.32, dA, iters);
  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 1; i++) {
    double t1 = dtime();
    kernel<<<blockCount, blockSize>>>((T)0.32, dA, iters);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  cudaFree(dA);

  double rcpThru = time.value() * gpu_clock * 1.0e6 / N / iters / warpCount;
  /*cout << setprecision(1) << fixed << typeid(T).name() << " " << setw(5) << N
       << " " << warpCount << " " << setw(5) << M << " "
       << " " << setw(5) << time.value() * 100 << " " << setw(5)
       << time.spread() * 100 << "%   " << setw(5) << setprecision(2) << rcpThru
       << "  " << setw(9) << clock << "MHz\n" ;*/
  return rcpThru;
}

template <typename T> void measureTabular(int maxWarpCount, const string& precision_name) {

  vector<map<pair<int, int>, double>> r(3);
  vector<string> operation_names = {"FMA_mixed", "DIV_separated", "SQRT_separated"};
  
  const int N = 1024;
  for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
    r[0][{warpCount, 1}] = measure<T, N, 1>(warpCount, FMA_mixed<T, N, 1>);
    r[1][{warpCount, 1}] =
        measure<T, N / 8, 1>(warpCount, DIV_separated<T, N / 8, 1>);
    r[2][{warpCount, 1}] =
        measure<T, N / 8, 1>(warpCount, SQRT_separated<T, N / 8, 1>);
    r[0][{warpCount, 2}] = measure<T, N, 2>(warpCount, FMA_mixed<T, N, 2>);
    r[1][{warpCount, 2}] =
        measure<T, N / 8, 2>(warpCount, DIV_separated<T, N / 8, 2>);
    r[2][{warpCount, 2}] =
        measure<T, N / 8, 2>(warpCount, SQRT_separated<T, N / 8, 2>);
    r[0][{warpCount, 4}] = measure<T, N, 4>(warpCount, FMA_mixed<T, N, 4>);
    r[1][{warpCount, 4}] =
        measure<T, N / 8, 4>(warpCount, DIV_separated<T, N / 8, 4>);
    r[2][{warpCount, 4}] =
        measure<T, N / 8, 4>(warpCount, SQRT_separated<T, N / 8, 4>);
    r[0][{warpCount, 8}] = measure<T, N, 8>(warpCount, FMA_mixed<T, N, 8>);
    r[1][{warpCount, 8}] =
        measure<T, N / 8, 8>(warpCount, DIV_separated<T, N / 8, 8>);
    r[2][{warpCount, 8}] =
        measure<T, N / 8, 8>(warpCount, SQRT_separated<T, N / 8, 8>);
  }

  // store results in global vector for SQL output
  for (int i = 0; i < 3; i++) {
    for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
      for (int streams = 1; streams <= 8; streams *= 2) {
        InCoreResult result;
        result.operation_name = operation_names[i];
        result.precision = precision_name;
        result.warp_count = warpCount;
        result.stream_count = streams;
        result.reciprocal_throughput = r[i][{warpCount, streams}];
        incore_results.push_back(result);
      }
    }
  }

  // print human-readable output only when not in SQL mode
  if (!sql_output_mode) {
    for (int i = 0; i < 3; i++) {
      for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
        for (int streams = 1; streams <= 8; streams *= 2) {
          cout << setw(7) << setprecision(3) << r[i][{warpCount, streams}] << " ";
        }
        cout << "\n";
      }
      cout << "\n";
    }
  }
}

int main(int argc, char **argv) {
  // parse execution_id from command line
  if (argc > 1) {
    execution_id = atoi(argv[1]);
    if (execution_id <= 0) {
      cerr << "Error: Invalid execution_id provided: " << argv[1] << endl;
      return 1;
    }
    sql_output_mode = true; // if execution_id provided, assume SQL mode
  }
  
  // parse gpu_uuid from command line
  if (argc > 2) {
    gpu_uuid = string(argv[2]);
    if (gpu_uuid.empty()) {
      cerr << "Error: Empty gpu_uuid provided" << endl;
      return 1;
    }
  }

  if (!sql_output_mode) {
    // only call getGPUClock() in interactive mode (it has console output)
    gpu_clock = getGPUClock();
  } else {
    // in SQL mode --> set a default value to avoid issues
    gpu_clock = 1980; // default GPU clock TODO
  }
  
  measureTabular<float>(32, "float");
  measureTabular<double>(32, "double");

  if (sql_output_mode) {
    // generate SQL INSERT statement
    cout << "-- CUDA In-Core Compute Benchmark Results\n";
    cout << "-- Generated at: " << time(nullptr) << "\n\n";
    
    if (!incore_results.empty()) {
      // group results by operation and precision
      map<pair<string, string>, vector<InCoreResult>> grouped_results;
      for (const auto& result : incore_results) {
        grouped_results[{result.operation_name, result.precision}].push_back(result);
      }
      
      // find best performance
      double best_throughput = 1e9; // start with high value (we want lowest reciprocal)
      string best_operation, best_precision;
      int optimal_warp_count = 0, optimal_stream_count = 0;
      
      for (const auto& result : incore_results) {
        if (result.reciprocal_throughput < best_throughput) {
          best_throughput = result.reciprocal_throughput;
          best_operation = result.operation_name;
          best_precision = result.precision;
          optimal_warp_count = result.warp_count;
          optimal_stream_count = result.stream_count;
        }
      }
      
      cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
      cout << "  " << execution_id << ", -- execution_id\n";
      cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
      cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'cuda_incore'),\n";
      cout << "  NOW(),\n";
      cout << "  'compute',\n";
      cout << "  'incore_performance',\n";
      cout << "  '{\n";
      cout << "    \"operation_types\": [\n";
      
      bool first_operation = true;
      for (const auto& group : grouped_results) {
        if (!first_operation) cout << ",\n";
        first_operation = false;
        
        cout << "      {\n";
        cout << "        \"operation_name\": \"" << group.first.first << "\",\n";
        cout << "        \"precision\": \"" << group.first.second << "\",\n";
        cout << "        \"configurations\": [\n";
        
        bool first_config = true;
        for (const auto& result : group.second) {
          if (!first_config) cout << ",\n";
          first_config = false;
          
          cout << "          {\n";
          cout << "            \"warp_count\": " << result.warp_count << ",\n";
          cout << "            \"stream_count\": " << result.stream_count << ",\n";
          cout << "            \"reciprocal_throughput\": " << fixed << setprecision(6) << result.reciprocal_throughput << "\n";
          cout << "          }";
        }
        cout << "\n        ]\n";
        cout << "      }";
      }
      
      cout << "\n    ],\n";
      cout << "    \"summary\": {\n";
      cout << "      \"total_configurations\": " << incore_results.size() << ",\n";
      cout << "      \"operation_count\": " << grouped_results.size() / 2 << ",\n"; // 3 operations
      cout << "      \"precision_count\": 2,\n";
      cout << "      \"best_performance\": {\n";
      cout << "        \"highest_throughput\": " << fixed << setprecision(6) << best_throughput << ",\n";
      cout << "        \"best_operation\": \"" << best_operation << "\",\n";
      cout << "        \"best_precision\": \"" << best_precision << "\",\n";
      cout << "        \"optimal_warp_count\": " << optimal_warp_count << ",\n";
      cout << "        \"optimal_stream_count\": " << optimal_stream_count << "\n";
      cout << "      }\n";
      cout << "    }\n";
      cout << "  }'::jsonb\n";
      cout << ");\n\n";
    }
  }
}
