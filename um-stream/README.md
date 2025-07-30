# CUDA Unified Memory STREAM Benchmark

A GPU unified memory bandwidth benchmark that measures memory performance using CUDA's unified memory system. This benchmark tests how efficiently the GPU can perform memory-intensive operations when using automatically managed, shared CPU-GPU memory rather than the usual explicit memory transfers.

This benchmark measures **unified memory performance** by running STREAM-like kernels on the automatically managed memory instead:

- Unified memory bandwidth for GPU compute operations
- Memory access scaling with different working set sizes  
- GPU memory subsystem performance without explicit transfers
- Automatic memory migration efficiency under compute load
- Sustained memory throughput for realistic workloads

## Unified Memory Operation

The benchmark is based on the original STREAM triad operation on unified memory instead:

```cuda
// Core unified memory kernel
__global__ void triad(double *A, double *B, double *C, size_t N) {
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = B[i] + C[i] * 1.3;  // read B, read C, write A
    }
}
```

| Component | Purpose |
|-----------|---------|
| **Unified Memory Arrays** | 3 arrays (A, B, C) using `cudaMallocManaged` |
| **Triad Operation** | Memory-intensive: 2 reads + 1 write per element |
| **Auto Migration** | System should manage CPU-GPU data movement automatically |
| **Working Set Scaling** | Test different memory pressure levels |

## Memory Access Pattern

- **Memory Type**: CUDA Unified Memory (automatically managed)
- **Operation**: `A[i] = B[i] + C[i] * 1.3` (STREAM triad)
- **Memory Streams**: 3 per operation (2 reads + 1 write)
- **Data Movement**: Automatic between CPU and GPU 

## How to Build and Run (Nvidia)

```bash
# Compile the benchmark
make

# Run the benchmark
./um-stream
```

## Output

```
 buffer size      time   spread   bandwidth
       24 MB     0.0ms    33.8%  2097.1GB/s
      192 MB     0.1ms     2.4%  2582.3GB/s
     1536 MB     0.6ms     0.6%  2919.8GB/s
    12288 MB     4.3ms     0.5%  2974.5GB/s
    49152 MB    17.3ms     0.1%  2974.1GB/s
```

**Column Descriptions:**
- **buffer size**: Total memory footprint for 3 arrays (A, B, C combined)
- **time**: Kernel execution time in milliseconds
- **spread**: Measurement variance as percentage
- **bandwidth**: Effective memory bandwidth in GB/s (accounts for 3 memory streams)
