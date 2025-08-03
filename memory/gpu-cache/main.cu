#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-clock.cuh"
#include "../../gpu-error.h"
#include "../../gpu-metrics/gpu-metrics.hpp"

#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

using namespace std;

#ifdef __NVCC__
using dtype = float;
#else
using dtype = float4;
#endif

dtype *dA, *dB;

struct CacheResult {
  double data_set_kb;
  double exec_time_ms;
  double spread_percent;
  double effective_bandwidth_gbps;
  double dram_read_gbps;
  double dram_write_gbps;
  double l2_read_gbps;
  double l2_store_gbps;
};

vector<CacheResult> cache_results;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (dtype)1.1;
  }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int zero) {
  dtype localSum = (dtype)0;

  B += threadIdx.x;

#pragma unroll N / BLOCKSIZE> 32   ? 1 : 32 / (N / BLOCKSIZE)
  for (int iter = 0; iter < iters; iter++) {
    B += zero;
    auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
    for (int i = 0; i < N; i += BLOCKSIZE) {
      localSum += B[i] * B2[i];
    }
    localSum *= (dtype)1.3;
  }
  if (localSum == (dtype)1233)
    A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
  return 0.0;
}

template <int N, int blockSize> void measure(bool sql_mode = false) {
  const size_t iters = (size_t)1000000000 / N + 2;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, iters, blockSize>, blockSize, 0));

  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  for (int i = 0; i < 15; i++) {
    const size_t bufferCount = 2 * N + i * 1282;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);

    dA += i;
    dB += i;

    GPU_ERROR(cudaDeviceSynchronize());

    GPU_ERROR(cudaEventRecord(start));
    callKernel<N, iters, blockSize>(blockCount);
    GPU_ERROR(cudaEventRecord(stop));

    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    time.add(milliseconds / 1000);

    measureDRAMBytesStart();
    callKernel<N, iters, blockSize>(blockCount);
    auto metrics = measureDRAMBytesStop();
    dram_read.add(metrics[0]);
    dram_write.add(metrics[1]);

    measureL2BytesStart();
    callKernel<N, iters, blockSize>(blockCount);
    metrics = measureL2BytesStop();
    L2_read.add(metrics[0]);
    L2_write.add(metrics[1]);

    GPU_ERROR(cudaFree(dA - i));
    GPU_ERROR(cudaFree(dB - i));
  }
  double blockDV = 2 * N * sizeof(dtype);

  double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
  
  if (sql_mode) {
    // store results for SQL output
    CacheResult result;
    result.data_set_kb = blockDV / 1024.0;
    result.exec_time_ms = time.value() * 1000.0;
    result.spread_percent = time.spread() * 100.0;
    result.effective_bandwidth_gbps = bw;
    result.dram_read_gbps = dram_read.value() / time.minValue() / 1.0e9;
    result.dram_write_gbps = dram_write.value() / time.minValue() / 1.0e9;
    result.l2_read_gbps = L2_read.value() / time.minValue() / 1.0e9;
    result.l2_store_gbps = L2_write.value() / time.minValue() / 1.0e9;
    cache_results.push_back(result);
  } else {
    // print human-readable output
    cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
         << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
         << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
         << setw(10) << bw << " GB/s"                                       //
         << setprecision(0) << setw(10)
         << dram_read.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << dram_write.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << L2_read.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl; //
  }
}

size_t constexpr expSeries(size_t N) {
  size_t val = 32 * 512;
  for (size_t i = 0; i < N; i++) {
    val *= 1.17;
  }
  return (val / 512) * 512;
}

int main(int argc, char **argv) {
  int execution_id = 1; // default fallback
  string gpu_uuid = "unknown_gpu"; // default fallback
  bool sql_output_mode = false;

  // parse execution_id from command line
  if (argc > 1) {
    execution_id = atoi(argv[1]);
    if (execution_id <= 0) {
      cerr << "Error: Invalid execution_id provided: " << argv[1] << endl;
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
  // unsigned int clock = getGPUClock();
  
  // only show header in interactive mode
  if (!sql_output_mode) {
    cout << setw(13) << "data set"   //
         << setw(12) << "exec time"  //
         << setw(11) << "spread"     //
         << setw(15) << "Eff. bw"    //
         << setw(16) << "DRAM read"  //
         << setw(16) << "DRAM write" //
         << setw(16) << "L2 read"    //
         << setw(16) << "L2 store\n";
  }

  initMeasureMetric();

  measure<128, 128>(sql_output_mode);
  measure<256, 256>(sql_output_mode);
  measure<512, 512>(sql_output_mode);
  measure<3 * 256, 256>(sql_output_mode);
  measure<2 * 512, 512>(sql_output_mode);
  measure<3 * 512, 512>(sql_output_mode);
  measure<4 * 512, 512>(sql_output_mode);
  measure<5 * 512, 512>(sql_output_mode);
  measure<6 * 512, 512>(sql_output_mode);
  measure<7 * 512, 512>(sql_output_mode);
  measure<8 * 512, 512>(sql_output_mode);
  measure<9 * 512, 512>(sql_output_mode);
  measure<10 * 512, 512>(sql_output_mode);
  measure<11 * 512, 512>(sql_output_mode);
  measure<12 * 512, 512>(sql_output_mode);
  measure<13 * 512, 512>(sql_output_mode);
  measure<14 * 512, 512>(sql_output_mode);
  measure<15 * 512, 512>(sql_output_mode);
  measure<16 * 512, 512>(sql_output_mode);
  measure<17 * 512, 512>(sql_output_mode);
  measure<18 * 512, 512>(sql_output_mode);
  measure<19 * 512, 512>(sql_output_mode);
  measure<20 * 512, 512>(sql_output_mode);
  measure<21 * 512, 512>(sql_output_mode);
  measure<22 * 512, 512>(sql_output_mode);
  measure<23 * 512, 512>(sql_output_mode);
  measure<24 * 512, 512>(sql_output_mode);
  measure<25 * 512, 512>(sql_output_mode);
  measure<26 * 512, 512>(sql_output_mode);
  measure<27 * 512, 512>(sql_output_mode);
  measure<28 * 512, 512>(sql_output_mode);
  measure<29 * 512, 512>(sql_output_mode);
  measure<30 * 512, 512>(sql_output_mode);
  measure<31 * 512, 512>(sql_output_mode);
  measure<32 * 512, 512>(sql_output_mode);

  measure<expSeries(1), 512>(sql_output_mode);
  measure<expSeries(2), 512>(sql_output_mode);
  measure<expSeries(3), 512>(sql_output_mode);
  measure<expSeries(4), 512>(sql_output_mode);
  measure<expSeries(5), 512>(sql_output_mode);
  measure<expSeries(6), 512>(sql_output_mode);
  measure<expSeries(7), 512>(sql_output_mode);
  measure<expSeries(8), 512>(sql_output_mode);
  measure<expSeries(9), 512>(sql_output_mode);
  measure<expSeries(10), 512>(sql_output_mode);
  measure<expSeries(11), 512>(sql_output_mode);
  measure<expSeries(12), 512>(sql_output_mode);
  measure<expSeries(13), 512>(sql_output_mode);
  measure<expSeries(14), 512>(sql_output_mode);
  measure<expSeries(16), 512>(sql_output_mode);
  measure<expSeries(17), 512>(sql_output_mode);
  measure<expSeries(18), 512>(sql_output_mode);
  measure<expSeries(19), 512>(sql_output_mode);
  measure<expSeries(20), 512>(sql_output_mode);
  measure<expSeries(21), 512>(sql_output_mode);
  measure<expSeries(22), 512>(sql_output_mode);
  measure<expSeries(23), 512>(sql_output_mode);
  measure<expSeries(24), 512>(sql_output_mode);
  measure<expSeries(25), 512>(sql_output_mode);
  measure<expSeries(26), 512>(sql_output_mode);
  measure<expSeries(27), 512>(sql_output_mode);
  measure<expSeries(28), 512>(sql_output_mode);
  measure<expSeries(29), 512>(sql_output_mode);
  measure<expSeries(30), 512>(sql_output_mode);
  measure<expSeries(31), 512>(sql_output_mode);
  measure<expSeries(32), 512>(sql_output_mode);
  measure<expSeries(33), 512>(sql_output_mode);
  measure<expSeries(34), 512>(sql_output_mode);
  measure<expSeries(35), 512>(sql_output_mode);
  measure<expSeries(36), 512>(sql_output_mode);
  measure<expSeries(37), 512>(sql_output_mode);
  measure<expSeries(38), 512>(sql_output_mode);
  measure<expSeries(39), 512>(sql_output_mode);
  measure<expSeries(40), 512>(sql_output_mode);
  measure<expSeries(41), 512>(sql_output_mode);
  measure<expSeries(42), 512>(sql_output_mode);
  measure<expSeries(43), 512>(sql_output_mode);
  measure<expSeries(44), 512>(sql_output_mode);
  measure<expSeries(45), 512>(sql_output_mode);
  measure<expSeries(46), 512>(sql_output_mode);
  measure<expSeries(47), 512>(sql_output_mode);
  measure<expSeries(48), 512>(sql_output_mode);
  measure<expSeries(49), 512>(sql_output_mode);

  // generate SQL output if in SQL mode
  if (sql_output_mode) {
    cout << "-- GPU Cache Benchmark Results\n";
    cout << "-- Generated at: " << time(nullptr) << "\n\n";
    
    if (!cache_results.empty()) {
      cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
      cout << "  " << execution_id << ", -- execution_id\n";
      cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
      cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'gpu_cache'),\n";
      cout << "  NOW(),\n";
      cout << "  'memory',\n";
      cout << "  'cache_bandwidth',\n";
      cout << "  '{\n";
      cout << "    \"configurations\": [\n";
      
      for (size_t i = 0; i < cache_results.size(); ++i) {
        const auto& result = cache_results[i];
        cout << "      {\n";
        cout << "        \"data_set_kb\": " << fixed << setprecision(1) << result.data_set_kb << ",\n";
        cout << "        \"exec_time_ms\": " << setprecision(1) << result.exec_time_ms << ",\n";
        cout << "        \"spread_percent\": " << setprecision(1) << result.spread_percent << ",\n";
        cout << "        \"effective_bandwidth_gbps\": " << setprecision(1) << result.effective_bandwidth_gbps << ",\n";
        cout << "        \"dram_read_gbps\": " << setprecision(1) << result.dram_read_gbps << ",\n";
        cout << "        \"dram_write_gbps\": " << setprecision(1) << result.dram_write_gbps << ",\n";
        cout << "        \"l2_read_gbps\": " << setprecision(1) << result.l2_read_gbps << ",\n";
        cout << "        \"l2_store_gbps\": " << setprecision(1) << result.l2_store_gbps << "\n";
        cout << "      }";
        if (i < cache_results.size() - 1) cout << ",";
        cout << "\n";
      }
      
      cout << "    ],\n";
      cout << "    \"total_configurations\": " << cache_results.size() << ",\n";
      cout << "    \"peak_performance\": {\n";
      
      // Find peak performance metrics
      double max_eff_bw = 0, max_l2_read = 0;
      double optimal_data_set = 0;
      for (const auto& result : cache_results) {
        if (result.effective_bandwidth_gbps > max_eff_bw) {
          max_eff_bw = result.effective_bandwidth_gbps;
          optimal_data_set = result.data_set_kb;
        }
        max_l2_read = max(max_l2_read, result.l2_read_gbps);
      }
      
      cout << "      \"max_effective_bandwidth_gbps\": " << setprecision(1) << max_eff_bw << ",\n";
      cout << "      \"max_l2_read_gbps\": " << setprecision(1) << max_l2_read << ",\n";
      cout << "      \"optimal_data_set_kb\": " << setprecision(1) << optimal_data_set << "\n";
      cout << "    }\n";
      cout << "  }'::jsonb\n";
      cout << ");\n\n";
    }
  }
}
