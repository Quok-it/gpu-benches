# CUDA Matrix Multiply Performance Benchmark

A comprehensive GPU matrix multiplication benchmark that measures TFLOPS performance across different precision types, matrix sizes, and Tensor Core utilization. This benchmark evaluates core GPU compute performance for that are primarily used for AI/ML workloads.

## Overview

This benchmark tests matrix multiply ops (A @ B) using various precisions and optimization techniques:

- **Regular cuBLAS operations**: FP64, FP32, FP16
- **Tensor Core accelerated**: FP16T, BFLOAT16T, TF32, INT8T
- **Matrix sizes**: 256×256 to 8192×8192 (square matrices)
- **Statistical accuracy**: Median of 100 measurements per config
- **Performance metric**: TFLOPS (Tera Floating-Point Ops Per Second)

## Precision Types Tested

| Precision | Description | Hardware Target | Tensor Cores |
|-----------|-------------|-----------------|--------------|
| **FP64** | Double-precision floating-point | Scientific HPC | No |
| **FP32** | Single-precision floating-point | Graphics, ML training | No |
| **FP16** | Half-precision floating-point | AI inference | No |
| **FP16T** | FP16 with Tensor Cores | AI training/inference | Yes |
| **BFLOAT16T** | Google's bfloat16 with Tensor Cores | AI training | Yes |
| **TF32** | TensorFloat-32 (NVIDIA Ampere+) | AI training | Yes |
| **INT8T** | 8-bit integer with Tensor Cores | AI inference | Yes |

## Build Instructions

```bash
# Compile the benchmark
make

# Run all matmul benchmarks
./cuda-matmul

# The benchmark will automatically:
# 1. Detect GPU capabilities
# 2. Run each precision type across all matrix sizes
# 3. Take 100 measurements per configuration
# 4. Report median TFLOPS performance
```

## Output Format

The benchmark produces several output sections:

### 1. System Info
```
CUDA Matrix Multiply Benchmark
Testing various precisions and tensor core usage
GPU Clock: 1980 1980 1980 ...
Device: NVIDIA H100 SXM5
Compute Capability: 8.9
Memory: 80 GB
Tensor Cores: Supported
```

### 2. Progressive Results
```
Running FP64 benchmarks...
  256x256: 2.35 TFLOPS
  512x512: 8.42 TFLOPS
  1024x1024: 15.67 TFLOPS
  ...
```

### 3. Detailed Results Table
```
=== BENCHMARK RESULTS ===
   Precision      TC    Size      TFLOPS    Time(ms)
----------------------------------------------------
        FP64      No     256        2.35       0.071
        FP64      No     512        8.42       0.126
        FP32     Yes    1024      156.84       0.034
        ...
```

### 4. TFLOPS Summary Matrix
```
=== TFLOPS SUMMARY BY MATRIX SIZE ===
   Precision     256     512    1024    2048    4096    8192
----------------------------------------------------------------
        FP64     2.4     8.4    15.7    18.2    19.1    19.3
        FP32     4.8    17.1    31.2    36.8    38.9    39.2
       FP16T    19.2    68.4   125.6   147.2   156.8   159.1
         ...
```

## References

- [NVIDIA cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Tensor Core Programming Guide](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
- [Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)