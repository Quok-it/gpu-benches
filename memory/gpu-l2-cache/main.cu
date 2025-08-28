#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-error.h"
#include "../../gpu-metrics/gpu-metrics.hpp"
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

using namespace std;

struct L2CacheResult {
  double data_set_kb;
  double buffer_size_kb;
  int block_run;
  double exec_time_ms;
  double timing_spread_percent;
  double effective_bandwidth_gbps;
  double dram_read_gbps;
  double dram_write_gbps;
  double l2_read_gbps;
  double l2_write_gbps;
};

vector<L2CacheResult> l2cache_results;

using dtype = double;
dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = dtype(1.1);
  }
}

template <int N, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int blockRun) {
  dtype localSum = dtype(0);

  for (int i = 0; i < N / 2; i++) {
    int idx =
        (blockDim.x * blockRun * i + (blockIdx.x % blockRun) * BLOCKSIZE) * 2 +
        threadIdx.x;
    localSum += B[idx] * B[idx + BLOCKSIZE];
  }

  localSum *= (dtype)1.3;
  if (threadIdx.x > 1233 || localSum == (dtype)23.12)
    A[threadIdx.x] += localSum;
}
template <int N, int blockSize>
double callKernel(int blockCount, int blockRun) {
  sumKernel<N, blockSize><<<blockCount, blockSize>>>(dA, dB, blockRun);
  GPU_ERROR(cudaPeekAtLastError());
  return 0.0;
}
template <int N> void measure(int blockRun, bool sql_mode = false) {

  const int blockSize = 1024;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, blockSize>, blockSize, 0));

  int blockCount = 200000;

  // GPU_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 11; i++) {
    const size_t bufferCount = blockRun * blockSize * N + i * 128;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    double t1 = dtime();
    callKernel<N, blockSize>(blockCount, blockRun);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);

    measureDRAMBytesStart();
    callKernel<N, blockSize>(blockCount, blockRun);
    auto metrics = measureDRAMBytesStop();
    dram_read.add(metrics[0]);
    dram_write.add(metrics[1]);

    measureL2BytesStart();
    callKernel<N, blockSize>(blockCount, blockRun);
    metrics = measureL2BytesStop();
    L2_read.add(metrics[0]);
    L2_write.add(metrics[1]);

    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }

  double blockDV = N * blockSize * sizeof(dtype);

  double bw = blockDV * blockCount / time.minValue() / 1.0e9;
  
  if (sql_mode) {
    // store results for SQL output
    L2CacheResult result;
    result.data_set_kb = blockDV / 1024.0;
    result.buffer_size_kb = blockDV * blockRun / 1024.0;
    result.block_run = blockRun;
    result.exec_time_ms = time.minValue() * 1000.0;
    result.timing_spread_percent = time.spread() * 100.0;
    result.effective_bandwidth_gbps = bw;
    result.dram_read_gbps = dram_read.median() / time.minValue() / 1.0e9;
    result.dram_write_gbps = dram_write.median() / time.minValue() / 1.0e9;
    result.l2_read_gbps = L2_read.median() / time.minValue() / 1.0e9;
    result.l2_write_gbps = L2_write.median() / time.minValue() / 1.0e9;
    l2cache_results.push_back(result);
  } else {
    // print human-readable output
    cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
         << fixed << setprecision(0) << setw(10) << blockDV * blockRun / 1024
         << " kB"                                                           //
         << setprecision(0) << setw(10) << time.minValue() * 1000.0 << "ms" //
         << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
         << setw(10) << bw << " GB/s   "                                    //
         << setprecision(0) << setw(6)
         << dram_read.median() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(6)
         << dram_write.median() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(6)
         << L2_read.median() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(6)
         << L2_write.median() / time.minValue() / 1.0e9 << " GB/s " << endl; //
  }
}

size_t constexpr expSeries(size_t N) {
  size_t val = 20;
  for (size_t i = 0; i < N; i++) {
    val = val * 1.04 + 1;
  }
  return val;
}

int main(int argc, char **argv) {
  string execution_id = "1"; // default fallback
  string gpu_uuid = "unknown_gpu"; // default fallback
  bool sql_output_mode = false;

  // parse execution_id from command line
  if (argc > 1) {
    execution_id = string(argv[1]);
    if (execution_id.empty()) {
      cerr << "Error: Empty execution_id provided" << endl;
      return 1;
    }
    sql_output_mode = true; // if args provided --> should be sql
  }

  // parse gpu_uuid from command line
  if (argc > 2) {
    gpu_uuid = string(argv[2]);
    if (gpu_uuid.empty()) {
      cerr << "Error: Empty gpu_uuid provided" << endl;
      return 1;
    }
  }

  initMeasureMetric();
  
  // only show header in interactive mode
  if (!sql_output_mode) {
    cout << setw(13) << "data set"   //
         << setw(12) << "exec time"  //
         << setw(11) << "spread"     //
         << setw(15) << "Eff. bw\n"; //
  }

  for (int i = 3; i < 10000; i += max(1.0, i * 0.1)) {
#ifdef __NVCC__
    measure<64>(i, sql_output_mode);
#else
    measure<64>(i, sql_output_mode);
#endif
  }

  // generate SQL output if in SQL mode
  if (sql_output_mode) {
    cout << "-- GPU L2 Cache Benchmark Results\n";
    cout << "-- Generated at: " << time(nullptr) << "\n\n";
    
    if (!l2cache_results.empty()) {
      cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
      cout << "  '" << execution_id << "', -- execution_id\n";
      cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
      cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'gpu_l2_cache'),\n";
      cout << "  NOW(),\n";
      cout << "  'memory',\n";
      cout << "  'l2_cache_bandwidth',\n";
      cout << "  '{\n";
      cout << "    \"configurations\": [\n";
      
      for (size_t j = 0; j < l2cache_results.size(); ++j) {
        const auto& result = l2cache_results[j];
        cout << "      {\n";
        cout << "        \"data_set_kb\": " << fixed << setprecision(1) << result.data_set_kb << ",\n";
        cout << "        \"buffer_size_kb\": " << setprecision(1) << result.buffer_size_kb << ",\n";
        cout << "        \"block_run\": " << result.block_run << ",\n";
        cout << "        \"exec_time_ms\": " << setprecision(1) << result.exec_time_ms << ",\n";
        cout << "        \"timing_spread_percent\": " << setprecision(1) << result.timing_spread_percent << ",\n";
        cout << "        \"effective_bandwidth_gbps\": " << setprecision(1) << result.effective_bandwidth_gbps << ",\n";
        cout << "        \"dram_read_gbps\": " << setprecision(1) << result.dram_read_gbps << ",\n";
        cout << "        \"dram_write_gbps\": " << setprecision(1) << result.dram_write_gbps << ",\n";
        cout << "        \"l2_read_gbps\": " << setprecision(1) << result.l2_read_gbps << ",\n";
        cout << "        \"l2_write_gbps\": " << setprecision(1) << result.l2_write_gbps << "\n";
        cout << "      }";
        if (j < l2cache_results.size() - 1) cout << ",";
        cout << "\n";
      }
      
      cout << "    ],\n";
      cout << "    \"total_configurations\": " << l2cache_results.size() << ",\n";
      cout << "    \"peak_performance\": {\n";
      
      // find peak performance metrics
      double max_eff_bw = 0, max_l2_read = 0;
      double optimal_buffer_size = 0;
      double min_latency = 1e9;
      for (const auto& result : l2cache_results) {
        if (result.effective_bandwidth_gbps > max_eff_bw) {
          max_eff_bw = result.effective_bandwidth_gbps;
          optimal_buffer_size = result.buffer_size_kb;
        }
        max_l2_read = max(max_l2_read, result.l2_read_gbps);
        min_latency = min(min_latency, result.exec_time_ms);
      }
      
      cout << "      \"max_effective_bandwidth_gbps\": " << setprecision(1) << max_eff_bw << ",\n";
      cout << "      \"max_l2_read_gbps\": " << setprecision(1) << max_l2_read << ",\n";
      cout << "      \"optimal_buffer_size_kb\": " << setprecision(1) << optimal_buffer_size << ",\n";
      cout << "      \"min_latency_ms\": " << setprecision(1) << min_latency << "\n";
      cout << "    }\n";
      cout << "  }'::jsonb\n";
      cout << ");\n\n";
    }
  }
}
