# CUDA In-Core Compute Benchmark

A GPU compute throughput characterization benchmark that measures the performance of different arithmetic operations across varying thread configurations. This benchmark helps understand GPU compute unit efficiency and optimal parallelism levels for compute-intensive workloads.

This benchmark measures **GPU arithmetic throughput** by running compute-intensive kernels with different operation types and measuring the reciprocal throughput (time per operation):

- Fused multiply-add (FMA) performance - should be the most efficient GPU operation
- Division operation performance - typically slower than FMA
- Square root operation performance - transcendental function performance  
- Thread parallelism scaling across warp (32 threads that execute tgt in lockstep of SIMT - single instructions, multiple thread) configurations
- Operation pipelining efficiency with different stream counts

## Compute Kernels

The benchmark includes 3 arithmetic operation types tested across different configurations:

| Kernel | Operation | Purpose | Compute Pattern |
|--------|-----------|---------|-----------------|
| **FMA_mixed** | `t = t * 0.9 + 0.5` | Fused multiply-add throughput | Mixed register operations |
| **DIV_separated** | `t = 0.1 / (t + 0.2)` | Division operation throughput | Sequential division operations |
| **SQRT_separated** | `t = sqrt(t + 0.2)` | Square root throughput | Transcendental function performance |

## Key Features

- **Warp Scaling**: Tests 1, 2, 4, 8, 16, 32 warps per block
- **Stream Parallelism**: Tests 1, 2, 4, 8 parallel computation streams per warp
- **Data Types**: Separate measurements for single (float) and double precision
- **Reciprocal Throughput**: Reports time per operation (lower means better performance)
- **Register Pressure**: Tests how register usage affects performance

## Computation Pattern

Each kernel performs intensive arithmetic in nested loops:

```cuda
// FMA kernel pattern
for (int iter = 0; iter < 10000; iter++) {
    T t[M];  // M parallel streams
    for (int m = 0; m < M; m++) {
        t[m] = initial_value;
    }
    for (int n = 0; n < N/M; n++) {
        for (int m = 0; m < M; m++) {
            t[m] = t[m] * 0.9 + 0.5;  // FMA operation
        }
    }
}
```

## How to Build and Run

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-incore
```

## Output Format

```
clock: 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 
   4.09    2.09     1.1     1.1 
   2.04    1.05   0.549   0.552 
   1.02   0.524   0.276   0.277 
  0.511   0.262   0.263   0.266 
  0.256   0.259   0.258   0.262 
  0.257   0.254   0.257   0.262 
```

### Output Structure:
The benchmark outputs **6 tables** in sequence:

1. **FMA_mixed (float)** - Fused multiply-add with single precision
2. **DIV_separated (float)** - Division operations with single precision  
3. **SQRT_separated (float)** - Square root operations with single precision
4. **FMA_mixed (double)** - Fused multiply-add with double precision
5. **DIV_separated (double)** - Division operations with double precision
6. **SQRT_separated (double)** - Square root operations with double precision

### Table Format:
- **Rows**: Warp counts (1, 2, 4, 8, 16, 32) - thread-level parallelism
- **Columns**: Stream counts (1, 2, 4, 8) - instruction-level parallelism  
- **Values**: Reciprocal throughput (cycles per operation)

### Column Explanation:
- **Column 1**: 1 stream per warp
- **Column 2**: 2 parallel streams per warp
- **Column 3**: 4 parallel streams per warp  
- **Column 4**: 8 parallel streams per warp