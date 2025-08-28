#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-error.h"
using namespace std;

struct MemcpyResult {
  int device_id;
  double transfer_size_kb;
  double exec_time_ms;
  double bandwidth_gbps;
  double timing_spread_percent;
};

vector<MemcpyResult> memcpy_results;

int main(int argc, char** argv) {
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

    int deviceCount;
    GPU_ERROR(cudaGetDeviceCount(&deviceCount));


    vector<char*> deviceBuffers(deviceCount, nullptr);
    char *host_buffer;
    const size_t buffer_size_bytes = (size_t)2 * 1024 * 1024 * 1024;


    for( int d  = 0; d < deviceCount; d++) {
        GPU_ERROR(cudaSetDevice(d));
        GPU_ERROR(cudaMalloc(& (deviceBuffers[d]), buffer_size_bytes));
        GPU_ERROR(cudaDeviceSynchronize());
    }
    GPU_ERROR(cudaMallocHost(&host_buffer, buffer_size_bytes));


    const int num_streams = 1;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    memset(host_buffer, 0, buffer_size_bytes);

    for (size_t transfer_size_bytes = 2 << 16;
       transfer_size_bytes <= buffer_size_bytes / num_streams;
       transfer_size_bytes *= 16) {

        for(int d = 0; d < deviceCount; d++) {
            GPU_ERROR(cudaSetDevice(d));
            MeasurementSeries time;
            for (int sample = 0; sample < 50; sample++) {
                memset(host_buffer, 0, buffer_size_bytes);
                double t1 = dtime();
                for (int stream = 0; stream < num_streams; stream++) {
                    GPU_ERROR(cudaMemcpyAsync(
                                  deviceBuffers[d] + (size_t)stream * transfer_size_bytes,
                                  host_buffer + (size_t)stream * transfer_size_bytes,
                                  transfer_size_bytes, cudaMemcpyDefault, streams[stream]));
                }

                GPU_ERROR(cudaDeviceSynchronize());
                double t2 = dtime();
                time.add(t2 - t1);
            }
            double bw = num_streams * transfer_size_bytes / time.value();
            
            if (sql_output_mode) {
                // store results for SQL output
                MemcpyResult result;
                result.device_id = d;
                result.transfer_size_kb = transfer_size_bytes >> 10;
                result.exec_time_ms = time.value() * 1000;
                result.bandwidth_gbps = bw * 1e-9;
                result.timing_spread_percent = time.spread() * 100;
                memcpy_results.push_back(result);
            } else {
                // print human-readable output
                cout << fixed  //
                    << "Device: " << d << "   "
                     << setw(10) << setprecision(0) << (transfer_size_bytes >> 10)
                     << "kB  "                                                      //
                     << setprecision(2) << setw(7) << time.value() * 1000 << "ms "  //
                     << setprecision(2) << setw(7) << bw * 1e-9 << "GB/s   "        //
                     << time.spread() * 100 << "%\n";
            }
        }
        if(deviceCount > 1 && !sql_output_mode) cout << "\n";
    }

    // generate SQL output if in SQL mode
    if (sql_output_mode) {
        cout << "-- CUDA Memory Copy Benchmark Results\n";
        cout << "-- Generated at: " << time(nullptr) << "\n\n";
        
        if (!memcpy_results.empty()) {
            cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
            cout << "  '" << execution_id << "', -- execution_id\n";
            cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
            cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'cuda_memcpy'),\n";
            cout << "  NOW(),\n";
            cout << "  'memory',\n";
            cout << "  'memcpy_bandwidth',\n";
            cout << "  '{\n";
            cout << "    \"configurations\": [\n";
            
            for (size_t i = 0; i < memcpy_results.size(); ++i) {
                const auto& result = memcpy_results[i];
                cout << "      {\n";
                cout << "        \"device_id\": " << result.device_id << ",\n";
                cout << "        \"transfer_size_kb\": " << fixed << setprecision(0) << result.transfer_size_kb << ",\n";
                cout << "        \"exec_time_ms\": " << setprecision(2) << result.exec_time_ms << ",\n";
                cout << "        \"bandwidth_gbps\": " << setprecision(2) << result.bandwidth_gbps << ",\n";
                cout << "        \"timing_spread_percent\": " << setprecision(2) << result.timing_spread_percent << "\n";
                cout << "      }";
                if (i < memcpy_results.size() - 1) cout << ",";
                cout << "\n";
            }
            
            cout << "    ],\n";
            cout << "    \"total_configurations\": " << memcpy_results.size() << ",\n";
            cout << "    \"peak_performance\": {\n";
            
            // find peak performance metrics
            double max_bandwidth = 0;
            double optimal_transfer_size = 0;
            double min_latency = 1e9;
            for (const auto& result : memcpy_results) {
                if (result.bandwidth_gbps > max_bandwidth) {
                    max_bandwidth = result.bandwidth_gbps;
                    optimal_transfer_size = result.transfer_size_kb;
                }
                min_latency = min(min_latency, result.exec_time_ms);
            }
            
            cout << "      \"max_bandwidth_gbps\": " << setprecision(2) << max_bandwidth << ",\n";
            cout << "      \"optimal_transfer_size_kb\": " << setprecision(0) << optimal_transfer_size << ",\n";
            cout << "      \"min_latency_ms\": " << setprecision(2) << min_latency << "\n";
            cout << "    }\n";
            cout << "  }'::jsonb\n";
            cout << ");\n\n";
        }
    }

    for(int d = 0; d< deviceCount; d++) {
        GPU_ERROR(cudaFree(deviceBuffers[d]));
}
    //  GPU_ERROR(cudaFree(host_buffer));
    GPU_ERROR(cudaFreeHost(host_buffer));
}
