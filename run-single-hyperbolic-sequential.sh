#!/bin/bash
set -e

# load env variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# get all GPU UUIDs
GPU_UUIDS=($(nvidia-smi -q | grep -i "GPU UUID" | awk '{print $NF}'))
if [ ${#GPU_UUIDS[@]} -eq 0 ]; then
    echo "Error: Could not get any GPU UUIDs"
    exit 1
fi

echo "Found ${#GPU_UUIDS[@]} GPUs:"
for i in "${!GPU_UUIDS[@]}"; do
    echo "  GPU $i: ${GPU_UUIDS[$i]}"
done

# set default directories if not provided
BENCHMARK_DIR=${BENCHMARK_DIR:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-$(pwd)/results}
# max number of GPUs to run in parallel (override with env var)
MAX_PARALLEL_GPUS=${MAX_PARALLEL_GPUS:-2}

cd "$BENCHMARK_DIR"

# create results directory
mkdir -p "$RESULTS_DIR/gpu-benches"

echo "Starting GPU benchmarks collection..."
echo "Results will be saved to: $RESULTS_DIR/gpu-benches"

# create new execution entry and get execution_id
echo "Creating new benchmark execution entry..."
EXECUTION_ID=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -q -c "
INSERT INTO benchmark_executions (execution_name, scale_type, benchmark_suite, provider, gpu_type_filter, status, started_at, total_gpus) 
VALUES ('Microbenchmarking Suite', 'gpu', 'microbenchmarks', 'Hyperbolic', 'unknown', 'running', NOW(), ${#GPU_UUIDS[@]}) 
RETURNING execution_id;" | xargs)

echo "Created execution with ID: $EXECUTION_ID"

# pre-build all benchmarks once to avoid compilation issues
echo ""
echo "================================================"
echo "======== Pre-building all benchmarks =========="
echo "================================================"

echo "Building gpu-stream..."
cd "$BENCHMARK_DIR/memory/gpu-stream"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: gpu-stream failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building gpu-cache..."
cd "$BENCHMARK_DIR/memory/gpu-cache"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: gpu-cache failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building gpu-l2-cache..."
cd "$BENCHMARK_DIR/memory/gpu-l2-cache"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: gpu-l2-cache failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building cuda-memcpy..."
cd "$BENCHMARK_DIR/memory/cuda-memcpy"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: cuda-memcpy failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building cuda-matmul..."
cd "$BENCHMARK_DIR/compute/cuda-matmul"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: cuda-matmul failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building cuda-incore..."
cd "$BENCHMARK_DIR/compute/cuda-incore"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: cuda-incore failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "Building gpu-small-kernels..."
cd "$BENCHMARK_DIR/system/gpu-small-kernels"
make clean && make
if [ $? -ne 0 ]; then echo "ERROR: gpu-small-kernels failed to build"; exit 1; fi
cd "$BENCHMARK_DIR"

echo "All benchmarks built successfully!"

# function to run benchmarks on a single GPU
run_benchmarks_on_gpu() {
    local gpu_index=$1
    local gpu_uuid=$2
    
    echo ""
    echo "================================================"
    echo "======== Running benchmarks on GPU $gpu_index ========"
    echo "========= UUID: $gpu_uuid ========="
    echo "================================================"
    
    # Set CUDA_VISIBLE_DEVICES to target specific GPU
    export CUDA_VISIBLE_DEVICES=$gpu_index
    
    echo "================================================"
    echo "======== Memory Microbenchmarks ================"
    echo "================================================"

    # gpu-stream benchmark
    echo ""
    echo "=== GPU Stream Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/memory/gpu-stream"
    ./cuda-stream "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/gpu-stream-results-gpu$gpu_index.sql"

    # insert into database
    echo "Inserting GPU Stream results into database..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-stream-results-gpu$gpu_index.sql"

    echo "GPU Stream microbenchmark completed and added to database!"
    cd "$BENCHMARK_DIR"

    # gpu-cache benchmark (needs sudo)
    echo ""
    echo "=== GPU Cache Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/memory/gpu-cache"
    if sudo -n true 2>/dev/null; then
        sudo ./cuda-cache "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/gpu-cache-results-gpu$gpu_index.sql"
        
        # insert into database
        echo "Inserting GPU Cache results into database..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-cache-results-gpu$gpu_index.sql"
        
        echo "GPU Cache microbenchmark completed and added to database!"
    else
        echo "GPU Cache microbenchmark skipped (requires sudo)"
    fi
    cd "$BENCHMARK_DIR"

    # gpu-l2-cache benchmark (needs sudo)
    echo ""
    echo "=== GPU L2 Cache Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/memory/gpu-l2-cache"
    if sudo -n true 2>/dev/null; then
        sudo ./cuda-l2-cache "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results-gpu$gpu_index.sql"
        
        # insert into database
        echo "Inserting GPU L2 Cache results into database..."
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results-gpu$gpu_index.sql"
        
        echo "GPU L2 Cache microbenchmark completed and added to database!"
    else
        echo "GPU L2 Cache microbenchmark skipped (requires sudo)"
    fi
    cd "$BENCHMARK_DIR"

    # cuda-memcpy benchmark
    echo ""
    echo "=== CUDA Memory Copy Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/memory/cuda-memcpy"
    ./cuda-memcpy "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/cuda-memcpy-results-gpu$gpu_index.sql"

    # insert into database
    echo "Inserting CUDA Memory Copy results into database..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-memcpy-results-gpu$gpu_index.sql"

    echo "CUDA Memory Copy microbenchmark completed and added to database!"
    cd "$BENCHMARK_DIR"

    echo "================================================"
    echo "======== Compute Microbenchmarks ==============="
    echo "================================================"

    # cuda-matmul benchmark
    echo ""
    echo "=== CUDA Matrix Multiplication Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/compute/cuda-matmul"
    ./cuda-matmul "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/cuda-matmul-results-gpu$gpu_index.sql"

    # insert into database
    echo "Inserting CUDA Matrix Multiplication results into database..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-matmul-results-gpu$gpu_index.sql"

    echo "CUDA Matrix Multiplication microbenchmark completed and added to database!"
    cd "$BENCHMARK_DIR"

    # cuda-incore benchmark
    echo ""
    echo "=== CUDA In-Core Compute Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/compute/cuda-incore"
    ./cuda-incore "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/cuda-incore-results-gpu$gpu_index.sql"

    # insert into database
    echo "Inserting CUDA In-Core results into database..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-incore-results-gpu$gpu_index.sql"

    echo "CUDA In-Core microbenchmark completed and added to database!"
    cd "$BENCHMARK_DIR"

    echo "================================================"
    echo "======== System Microbenchmarks ================"
    echo "================================================"

    # gpu-small-kernels benchmark
    echo ""
    echo "=== GPU Small Kernels System Microbenchmark (GPU $gpu_index) ==="
    cd "$BENCHMARK_DIR/system/gpu-small-kernels"
    ./cuda-small-kernels "$EXECUTION_ID" "$gpu_uuid" > "$RESULTS_DIR/gpu-benches/gpu-small-kernels-results-gpu$gpu_index.sql"

    # insert into database
    echo "Inserting GPU Small Kernels results into database..."
    PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-small-kernels-results-gpu$gpu_index.sql"

    echo "GPU Small Kernels microbenchmark completed and added to database!"
    cd "$BENCHMARK_DIR"
    
    echo "Completed all benchmarks for GPU $gpu_index"
}

# helper function: wait until fewer than MAX_PARALLEL_GPUS background jobs are running
wait_for_available_slot() {
    # maintain a list of active PIDs in the array 'PIDS'
    while [ ${#PIDS[@]} -ge "$MAX_PARALLEL_GPUS" ]; do
        # prune finished PIDs
        for idx in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$idx]}" 2>/dev/null; then
                unset 'PIDS[$idx]'
            fi
        done
        # reindex array to close gaps
        PIDS=("${PIDS[@]}")
        if [ ${#PIDS[@]} -ge "$MAX_PARALLEL_GPUS" ]; then
            sleep 1
        fi
    done
}

# run benchmarks on all GPUs with bounded parallelism
echo ""
echo "Starting parallel multi-GPU benchmark execution..."
echo "Running benchmarks on all ${#GPU_UUIDS[@]} GPUs with up to $MAX_PARALLEL_GPUS in parallel..."

declare -a PIDS=()

for i in "${!GPU_UUIDS[@]}"; do
    wait_for_available_slot
    echo ""
    echo "********************************************************"
    echo "Launching benchmarks on GPU $i (background)"
    echo "********************************************************"
    run_benchmarks_on_gpu "$i" "${GPU_UUIDS[$i]}" &
    PIDS+=($!)
done

# wait for all background jobs to finish
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

echo ""
echo "********************************************************"
echo "All GPU benchmarks completed!"
echo "********************************************************"

# mark execution as completed
echo ""
echo "Marking execution as completed in database..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
UPDATE benchmark_executions 
SET status = 'completed', completed_at = NOW() 
WHERE execution_id = $EXECUTION_ID;"

# Copy README files for reference
echo ""
echo "Copying documentation files..."
find . -name "README.md" -exec cp {} "$RESULTS_DIR/gpu-benches/" \; 2>/dev/null || true

echo ""
echo "All GPU benchmarks completed successfully on ${#GPU_UUIDS[@]} GPUs!"
echo "Results saved to: $RESULTS_DIR/gpu-benches/"
echo "Result files are named with GPU index suffixes (e.g., *-gpu0.sql, *-gpu1.sql, etc.)"