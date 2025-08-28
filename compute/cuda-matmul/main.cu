#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-clock.cuh"
#include "../../gpu-error.h"
#include "../../metrics.cuh"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>

using namespace std;

// global variables
cublasHandle_t cublas_handle;
cublasLtHandle_t cublaslt_handle;
unsigned int gpu_clock = 0;

// SQL mode globals
string execution_id = "1";
string gpu_uuid = "unknown_gpu";
bool sql_output_mode = false;

// matrix dimensions to test
vector<int> matrix_sizes = {256, 512, 1024, 2048, 4096, 8192};

// precision types
enum PrecisionType {
    PREC_FP64, // most accurate but slowest (scientific computing)
    PREC_FP32, // standard floating point
    PREC_FP16, // half precision (common in AI)
    PREC_BF16, // bfloat16 (Google's AI-optimized precision)
    PREC_TF32, // tensor float (NVIDIA's AI-optimized precision - newer GPUs)
    PREC_INT8, // 8-bit integer (common in AI/Inference)
    PREC_FP64_TENSOR, // Scientific computing (Tensor Cores - specialized for speed boost)
    PREC_FP32_TENSOR, // Standard floating point (Tensor Cores)
    PREC_FP16_TENSOR, // Half precision (Tensor Cores)
    PREC_BF16_TENSOR, // Bfloat16 (Tensor Cores)
    PREC_INT8_TENSOR // Inference (Tensor Cores)
};

struct BenchmarkResult {
    double tflops;
    double time_ms;
    int matrix_size;
    string precision_name;
    bool tensor_cores;
    bool sparse;
};

// helper function to get FLOPS for matrix multiply
double get_flops(int M, int N, int K) {
    // times 2 because multiply-add
    return 2.0 * M * N * K; // count total math ops
}

// init matrices with random data
template<typename T>
void init_matrix(T* matrix, int size, T value = T(1.0)) {
    for(int i = 0; i < size; i++) {
        matrix[i] = value + T(i % 100) / T(100.0);
    }
}

// FP64 matrix multiply using cuBLAS
double benchmark_fp64_cublas(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(double);
    size_t size_b = K * N * sizeof(double);
    size_t size_c = M * N * sizeof(double);
    
    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (double*)malloc(size_a);
    h_b = (double*)malloc(size_b);
    h_c = (double*)malloc(size_c);
    
    // init matrices
    init_matrix(h_a, M * K);
    init_matrix(h_b, K * N);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    const double alpha = 1.0, beta = 0.0;
    
    // warmup to remove startup overhead from calcs
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0); // convert to ms
    }
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// FP32 matrix multiply using cuBLAS
double benchmark_fp32_cublas(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    
    // init matrices
    init_matrix(h_a, M * K);
    init_matrix(h_b, K * N);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // warmup to remove startup overhead from calcs
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// FP16 matrix multiply using cuBLAS
double benchmark_fp16_cublas(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(__half);
    size_t size_b = K * N * sizeof(__half);
    size_t size_c = M * N * sizeof(__half);
    
    __half *h_a, *h_b, *h_c;
    __half *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (__half*)malloc(size_a);
    h_b = (__half*)malloc(size_b);
    h_c = (__half*)malloc(size_c);
    
    // init matrices
    for(int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f + (i % 100) / 100.0f);
    for(int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f + (i % 100) / 100.0f);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    // warmup to remove startup overhead from calcs
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// BF16 matrix multiply using cublasLt (Tensor Cores) - too new fpr regular cublas
double benchmark_bf16_tensor(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(__nv_bfloat16);
    size_t size_b = K * N * sizeof(__nv_bfloat16);
    size_t size_c = M * N * sizeof(__nv_bfloat16);
    
    __nv_bfloat16 *h_a, *h_b, *h_c;
    __nv_bfloat16 *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (__nv_bfloat16*)malloc(size_a);
    h_b = (__nv_bfloat16*)malloc(size_b);
    h_c = (__nv_bfloat16*)malloc(size_c);
    
    // init matrices
    for(int i = 0; i < M * K; i++) h_a[i] = __float2bfloat16(1.0f + (i % 100) / 100.0f);
    for(int i = 0; i < K * N; i++) h_b[i] = __float2bfloat16(1.0f + (i % 100) / 100.0f);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // setup cublasLt operation (newer GPUs)
    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
    
    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, M, K, M);
    cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, M, N, M);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // warmup to remove startup overhead from calcs
    cublasLtMatmul(cublaslt_handle, matmul_desc,
                   &alpha, d_a, a_desc, d_b, b_desc,
                   &beta, d_c, c_desc, d_c, c_desc,
                   nullptr, nullptr, 0, 0);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasLtMatmul(cublaslt_handle, matmul_desc,
                       &alpha, d_a, a_desc, d_b, b_desc,
                       &beta, d_c, c_desc, d_c, c_desc,
                       nullptr, nullptr, 0, 0);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // cleanup
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// TF32 matrix multiply (Tensor Cores on Ampere+)
double benchmark_tf32_tensor(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c = (float*)malloc(size_c);
    
    // init matrices
    init_matrix(h_a, M * K);
    init_matrix(h_b, K * N);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    // enable TF32 for tensor operations
    cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // warmup to remove startup overhead from calcs
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha,
                 d_b, CUDA_R_32F, N,
                 d_a, CUDA_R_32F, K, &beta,
                 d_c, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     d_b, CUDA_R_32F, N,
                     d_a, CUDA_R_32F, K, &beta,
                     d_c, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // reset math mode
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// INT8 matrix multiply using Tensor Cores
double benchmark_int8_tensor(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(int8_t);
    size_t size_b = K * N * sizeof(int8_t);
    size_t size_c = M * N * sizeof(int32_t);
    
    int8_t *h_a, *h_b;
    int32_t *h_c;
    int8_t *d_a, *d_b;
    int32_t *d_c;
    
    // allocate host memory
    h_a = (int8_t*)malloc(size_a);
    h_b = (int8_t*)malloc(size_b);
    h_c = (int32_t*)malloc(size_c);
    
    // init matrices
    for(int i = 0; i < M * K; i++) h_a[i] = (int8_t)(i % 127);
    for(int i = 0; i < K * N; i++) h_b[i] = (int8_t)(i % 127);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    const int32_t alpha = 1, beta = 0;
    
    // warmup to remove startup overhead from calcs
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha,
                 d_b, CUDA_R_8I, N,
                 d_a, CUDA_R_8I, K, &beta,
                 d_c, CUDA_R_32I, N,
                 CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     d_b, CUDA_R_8I, N,
                     d_a, CUDA_R_8I, K, &beta,
                     d_c, CUDA_R_32I, N,
                     CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// FP16 with Tensor Cores
double benchmark_fp16_tensor(int M, int N, int K, int num_iterations = 100) {
    size_t size_a = M * K * sizeof(__half);
    size_t size_b = K * N * sizeof(__half);
    size_t size_c = M * N * sizeof(__half);
    
    __half *h_a, *h_b, *h_c;
    __half *d_a, *d_b, *d_c;
    
    // allocate host memory
    h_a = (__half*)malloc(size_a);
    h_b = (__half*)malloc(size_b);
    h_c = (__half*)malloc(size_c);
    
    // init matrices
    for(int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f + (i % 100) / 100.0f);
    for(int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f + (i % 100) / 100.0f);
    
    // allocate device memory
    GPU_ERROR(cudaMalloc(&d_a, size_a));
    GPU_ERROR(cudaMalloc(&d_b, size_b));
    GPU_ERROR(cudaMalloc(&d_c, size_c));
    
    // copy to device
    GPU_ERROR(cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice));
    GPU_ERROR(cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice));
    
    const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    
    // enable tensor operations
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    
    // warmup to remove startup overhead from calcs
    cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, &alpha,
                 d_b, CUDA_R_16F, N,
                 d_a, CUDA_R_16F, K, &beta,
                 d_c, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    GPU_ERROR(cudaDeviceSynchronize());
    
    // benchmark
    MeasurementSeries times;
    for(int i = 0; i < num_iterations; i++) {
        double start = dtime();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     d_b, CUDA_R_16F, N,
                     d_a, CUDA_R_16F, K, &beta,
                     d_c, CUDA_R_16F, N,
                     CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        GPU_ERROR(cudaDeviceSynchronize());
        double end = dtime();
        times.add((end - start) * 1000.0);
    }
    
    // reset math mode
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    
    // cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    
    return times.median();
}

// main benchmark runner
void run_all_benchmarks() {
    vector<BenchmarkResult> results;
    
    if (!sql_output_mode) {
        cout << "Starting Matrix Multiply Benchmarks..." << endl;
        cout << "Matrix Sizes: ";
        for(int size : matrix_sizes) cout << size << " ";
        cout << endl << endl;
    }
    
    // define benchmark configs
    struct BenchConfig {
        string name;
        PrecisionType type;
        bool tensor_cores;
        double (*func)(int, int, int, int);
    };
    
    vector<BenchConfig> benchmarks = {
        {"FP64", PREC_FP64, false, benchmark_fp64_cublas},
        {"FP32", PREC_FP32, false, benchmark_fp32_cublas},
        {"FP16", PREC_FP16, false, benchmark_fp16_cublas},
        {"FP16T", PREC_FP16_TENSOR, true, benchmark_fp16_tensor},
        {"BF16T", PREC_BF16_TENSOR, true, benchmark_bf16_tensor},
        {"TF32", PREC_TF32, true, benchmark_tf32_tensor},
        {"INT8T", PREC_INT8_TENSOR, true, benchmark_int8_tensor}
    };
    
    for(const auto& bench : benchmarks) {
        if (!sql_output_mode) {
            cout << "Running " << bench.name << " benchmarks..." << endl;
        }
        
        for(int size : matrix_sizes) {
            try {
                double time_ms = bench.func(size, size, size, 100);
                double flops = get_flops(size, size, size);
                double tflops = (flops / (time_ms / 1000.0)) / 1e12;
                
                BenchmarkResult result;
                result.tflops = tflops;
                result.time_ms = time_ms;
                result.matrix_size = size;
                result.precision_name = bench.name;
                result.tensor_cores = bench.tensor_cores;
                result.sparse = false;
                
                results.push_back(result);
                
                if (!sql_output_mode) {
                    cout << "  " << size << "x" << size << ": " 
                         << fixed << setprecision(2) << tflops << " TFLOPS" << endl;
                }
            } catch(...) {
                if (!sql_output_mode) {
                    cout << "  " << size << "x" << size << ": FAILED" << endl;
                }
            }
        }
        if (!sql_output_mode) {
            cout << endl;
        }
    }
    
    if (sql_output_mode) {
        // generate SQL output
        cout << "-- CUDA Matrix Multiplication Benchmark Results\n";
        cout << "-- Generated at: " << time(nullptr) << "\n\n";
        
        if (!results.empty()) {
            cout << "INSERT INTO gpu_scale_results (execution_id, gpu_uuid, benchmark_id, timestamp, test_category, test_name, results) VALUES (\n";
            cout << "  '" << execution_id << "', -- execution_id\n";
            cout << "  '" << gpu_uuid << "', -- gpu_uuid\n";
            cout << "  (SELECT benchmark_id FROM benchmark_definitions WHERE benchmark_name = 'cuda_matmul'),\n";
            cout << "  NOW(),\n";
            cout << "  'compute',\n";
            cout << "  'matrix_multiplication_tflops',\n";
            cout << "  '{\n";
            cout << "    \"configurations\": [\n";
            
            for (size_t i = 0; i < results.size(); ++i) {
                const auto& result = results[i];
                cout << "      {\n";
                cout << "        \"precision_type\": \"" << result.precision_name << "\",\n";
                cout << "        \"matrix_size\": " << result.matrix_size << ",\n";
                cout << "        \"tensor_cores\": " << (result.tensor_cores ? "true" : "false") << ",\n";
                cout << "        \"tflops\": " << fixed << setprecision(2) << result.tflops << ",\n";
                cout << "        \"time_ms\": " << setprecision(3) << result.time_ms << ",\n";
                cout << "        \"flops\": " << scientific << setprecision(2) << get_flops(result.matrix_size, result.matrix_size, result.matrix_size) << "\n";
                cout << "      }";
                if (i < results.size() - 1) cout << ",";
                cout << "\n";
            }
            
            cout << "    ],\n";
            cout << "    \"total_configurations\": " << results.size() << ",\n";
            cout << "    \"peak_performance\": {\n";
            
            // find peak performance metrics
            double max_tflops = 0, tensor_peak = 0, traditional_peak = 0;
            string best_precision;
            int best_size = 0;
            for (const auto& result : results) {
                if (result.tflops > max_tflops) {
                    max_tflops = result.tflops;
                    best_precision = result.precision_name;
                    best_size = result.matrix_size;
                }
                if (result.tensor_cores) {
                    tensor_peak = max(tensor_peak, result.tflops);
                } else {
                    traditional_peak = max(traditional_peak, result.tflops);
                }
            }
            
            cout << "      \"max_tflops\": " << fixed << setprecision(2) << max_tflops << ",\n";
            cout << "      \"best_precision\": \"" << best_precision << "\",\n";
            cout << "      \"best_matrix_size\": " << best_size << ",\n";
            cout << "      \"tensor_core_peak\": " << setprecision(2) << tensor_peak << ",\n";
            cout << "      \"traditional_peak\": " << setprecision(2) << traditional_peak << "\n";
            cout << "    },\n";
            cout << "    \"precision_summary\": {\n";
            
            // create precision summary
            bool first_precision = true;
            for (const auto& bench : benchmarks) {
                if (!first_precision) cout << ",\n";
                cout << "      \"" << bench.name << "\": {\n";
                cout << "        \"tensor_cores\": " << (bench.tensor_cores ? "true" : "false") << ",\n";
                cout << "        \"results_by_size\": {\n";
                
                bool first_size = true;
                for (int size : matrix_sizes) {
                    for (const auto& result : results) {
                        if (result.precision_name == bench.name && result.matrix_size == size) {
                            if (!first_size) cout << ",\n";
                            cout << "          \"" << size << "\": " << fixed << setprecision(1) << result.tflops;
                            first_size = false;
                            break;
                        }
                    }
                }
                cout << "\n        }\n";
                cout << "      }";
                first_precision = false;
            }
            
            cout << "\n    }\n";
            cout << "  }'::jsonb\n";
            cout << ");\n\n";
        }
    } else {
        // print results table
        cout << "\n=== BENCHMARK RESULTS ===" << endl;
        cout << setw(12) << "Precision" << setw(8) << "TC" << setw(8) << "Size";
        cout << setw(12) << "TFLOPS" << setw(12) << "Time(ms)" << endl;
        cout << string(52, '-') << endl;
        
        for(const auto& result : results) {
            cout << setw(12) << result.precision_name 
                 << setw(8) << (result.tensor_cores ? "Yes" : "No")
                 << setw(8) << result.matrix_size
                 << setw(12) << fixed << setprecision(2) << result.tflops
                 << setw(12) << fixed << setprecision(3) << result.time_ms << endl;
        }
        
        // print summary table by precision
        cout << "\n=== TFLOPS SUMMARY BY MATRIX SIZE ===" << endl;
        cout << setw(12) << "Precision";
        for(int size : matrix_sizes) {
            cout << setw(8) << size;
        }
        cout << endl;
        cout << string(12 + 8 * matrix_sizes.size(), '-') << endl;
        
        for(const auto& bench : benchmarks) {
            cout << setw(12) << bench.name;
            for(int size : matrix_sizes) {
                bool found = false;
                for(const auto& result : results) {
                    if(result.precision_name == bench.name && result.matrix_size == size) {
                        cout << setw(8) << fixed << setprecision(1) << result.tflops;
                        found = true;
                        break;
                    }
                }
                if(!found) cout << setw(8) << "N/A";
            }
            cout << endl;
        }
    }
}

int main(int argc, char **argv) {
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

    if (!sql_output_mode) {
        cout << "CUDA Matrix Multiply Benchmark" << endl;
        cout << "Testing various precisions and tensor core usage" << endl;
        cout << "GPU Clock: ";
        
        // init GPU clock measurement (only in interactive mode)
        gpu_clock = getGPUClock();
    }
    
    // init cuBLAS
    cublasCreate(&cublas_handle);
    cublasLtCreate(&cublaslt_handle);
    
    // get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (!sql_output_mode) {
        cout << "Device: " << prop.name << endl;
        cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
        cout << "Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << endl;
        
        // check for tensor core support
        bool has_tensor_cores = (prop.major >= 7); // Volta and newer
        cout << "Tensor Cores: " << (has_tensor_cores ? "Supported" : "Not Supported") << endl;
        cout << endl;
        
        if(!has_tensor_cores) {
            cout << "Warning: Tensor Core benchmarks may not run optimally on this device." << endl;
        }
    }
    
    // run benchmarks
    run_all_benchmarks();
    
    // cleanup
    cublasDestroy(cublas_handle);
    cublasLtDestroy(cublaslt_handle);
    
    return 0;
}