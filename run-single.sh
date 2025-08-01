#!/bin/bash
set -e

# set default directories if not provided
BENCHMARK_DIR=${BENCHMARK_DIR:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-$(pwd)/results}

cd "$BENCHMARK_DIR"

# create results directory
mkdir -p "$RESULTS_DIR/gpu-benches"

# TODO: convert this to prometheus exporter rather than txt file
echo "Starting GPU benchmarks collection..."
echo "Results will be saved to: $RESULTS_DIR/gpu-benches"

# gpu-stream benchmark
echo ""
echo "=== GPU Stream Microbenchmark ==="
cd memory/gpu-stream
make clean && make
./cuda-stream > "$RESULTS_DIR/gpu-benches/gpu-stream-results.txt"
echo "GPU Stream microbenchmark completed"
cd ../..

# gpu-cache benchmark (needs sudo)
echo ""
echo "=== GPU Cache Microbenchmark ==="
cd memory/gpu-cache
make clean && make
if sudo -n true 2>/dev/null; then
    sudo ./cuda-cache > "$RESULTS_DIR/gpu-benches/gpu-cache-results.txt"
fi
echo "GPU Cache microbenchmark completed"
cd ../..

# gpu-l2-cache benchmark (needs sudo)
echo ""
echo "=== GPU L2 Cache Microbenchmark ==="
cd memory/gpu-l2-cache
make clean && make
if sudo -n true 2>/dev/null; then
    sudo ./cuda-l2-cache > "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results.txt"
fi
echo "GPU L2 Cache microbenchmark completed"
cd ../..

# cuda-memcpy benchmark
echo ""
echo "=== CUDA Memory Copy Microbenchmark ==="
cd memory/cuda-memcpy
make clean && make
./cuda-memcpy > "$RESULTS_DIR/gpu-benches/cuda-memcpy-results.txt"
echo "CUDA Memory Copy microbenchmark completed"
cd ../..

# Copy README files for reference
echo ""
echo "Copying documentation files..."
find . -name "README.md" -exec cp {} "$RESULTS_DIR/gpu-benches/" \; 2>/dev/null || true

echo ""
echo "All GPU benchmarks completed successfully!"
echo "Results saved to: $RESULTS_DIR/gpu-benches/"