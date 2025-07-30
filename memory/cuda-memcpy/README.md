# CUDA Memory Copy Benchmark

A GPU memory transfer benchmark that measures CPU-GPU interconnect bandwidth between host and device memory across different transfer sizes. This benchmark characterizes the performance of data movement over the CPU-GPU interconnect (PCIe, SXM, or other high-speed links).

This benchmark measures **CPU-to-GPU memory transfer performance** by testing how efficiently data moves between host and device memory:

- CPU-GPU interconnect bandwidth across different transfer sizes (128 kB to 2 GB)
- Memory copy scaling behaviour and optimal transfer sizes
- Asynchronous transfer efficiency using CUDA streams
- This simulates real-world memory movement patterns
- Multi-GPU memory transfer performance comparison (possible)

## Memory Copy Operation

The benchmark uses asynchronous CUDA memory copies to test CPU-GPU interconnect performance:

```cuda
// Core memory copy operation
cudaMemcpyAsync(device_buffer, host_buffer, transfer_size, 
                cudaMemcpyDefault, stream);
```

| Component | Purpose |
|-----------|---------|
| **Host Buffer** | 2 GB pinned memory on CPU (fast interconnect access / GPU transfer) |
| **Device Buffer** | 2 GB GPU memory on each available GPU to receive data |
| **Transfer Sizes** | 128 kB to 2 GB (scaling by factors of 16) |
| **Async Copies** | Non-blocking transfers using CUDA streams (fire and forget tests realistic performance) |

## Memory Transfer Pattern

- **Host Memory**: Pinned (page-locked) for optimal interconnect performance (locked in place means GPU can access directly so faster)
- **Transfer Direction**: CPU → GPU (Host-to-Device)
- **Transfer Method**: Asynchronous CUDA memory copy (CPU can do other work while copy happens)
- **Scaling**: Progressive transfer sizes from small to large

## How to Build and Run (Nvidia)

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-memcpy
```

## Output

```
Device: 0          128kB     0.03ms    4.02GB/s   115.61%
Device: 0         2048kB     0.07ms   29.13GB/s   27.81%
Device: 0        32768kB     0.76ms   44.11GB/s   1.57%
Device: 0       524288kB    11.65ms   46.08GB/s   1.81%
```

**Column Descriptions:**
- **Device**: GPU device ID (0, 1, 2... for multi-GPU systems - potentially)
- **Transfer Size**: Amount of data copied in kB
- **Transfer Time**: Time to complete the memory copy in milliseconds
- **Bandwidth**: Effective interconnect bandwidth in GB/s
- **Spread**: Measurement variance as percentage

