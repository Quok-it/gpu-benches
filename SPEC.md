# GPU Benchmarks Collection Specification

## Overview
This collection contains GPU memory hierarchy benchmarks that test different aspects of GPU memory performance, from cache behavior to CPU-GPU data transfers.

## Benchmarks Included

### gpu-cache - GPU Cache Hierarchy Benchmark
Tests GPU memory hierarchy (L1/L2 cache, DRAM) performance with varying working set sizes.

**Output Fields:**
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| data_set | integer | kB | Working set size per SM |
| exec_time | float | ms | Kernel execution time |
| spread | float | % | Measurement variance |
| eff_bw | float | GB/s | Effective bandwidth |
| dram_read | float | GB/s | DRAM read bandwidth (requires sudo) |
| dram_write | float | GB/s | DRAM write bandwidth (requires sudo) |
| l2_read | float | GB/s | L2 cache read bandwidth (requires sudo) |
| l2_store | float | GB/s | L2 cache store bandwidth (requires sudo) |

### gpu-l2-cache - GPU L2 Cache Contention Benchmark  
Tests L2 cache performance under multi-block contention with fixed per-block working sets.

**Output Fields:**
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| per_block_size | integer | kB | Fixed working set per thread block (512 kB) |
| total_size | integer | kB | Total memory footprint across all blocks |
| exec_time | float | ms | Kernel execution time |
| spread | float | % | Measurement variance |
| eff_bw | float | GB/s | Effective bandwidth |

### cuda-memcpy - CPU-GPU Transfer Benchmark
Tests CPU-GPU interconnect bandwidth across different transfer sizes.

**Output Fields:**
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| device_id | integer | - | GPU device identifier |
| transfer_size | integer | kB | Amount of data transferred |
| transfer_time | float | ms | Time to complete transfer |
| bandwidth | float | GB/s | Effective interconnect bandwidth |
| spread | float | % | Measurement variance |

### um-stream - Unified Memory Stream Benchmark
Tests unified memory performance with automatic CPU-GPU memory management.

**Output Fields:**
| Field | Type | Unit | Description |
|-------|------|------|-------------|
| buffer_size | integer | MB | Total memory footprint (3 arrays combined) |
| exec_time | float | ms | Kernel execution time |
| spread | float | % | Measurement variance |
| bandwidth | float | GB/s | Effective memory bandwidth |

## Requirements
- NVIDIA GPU with CUDA support
- CUDA toolkit and runtime
- Root access for performance counters (gpu-cache, gpu-l2-cache)
- CUPTI libraries for detailed metrics
- Sufficient system memory for unified memory tests

## Output Files
- `gpu-cache-results.txt`: Cache hierarchy performance data
- `gpu-l2-cache-results.txt`: L2 cache contention analysis
- `cuda-memcpy-results.txt`: CPU-GPU transfer performance
- `um-stream-results.txt`: Unified memory bandwidth data
- Various `README.md` files: Documentation for each benchmark