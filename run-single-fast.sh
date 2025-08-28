# runs gpu-stream, cuda-memcpy, cuda-matmul, cuda-incore on all available GPUs in parallel

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

echo "Found ${#GPU_UUIDS[@]} GPU(s):"
for i in "${!GPU_UUIDS[@]}"; do
    echo "  GPU $i: ${GPU_UUIDS[$i]}"
done

# get execution ID from env var or latest from db
if [ -z "$EXECUTION_ID" ]; then
    echo "EXECUTION_ID not set, fetching latest from db..."
    EXECUTION_ID=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
    SELECT execution_id 
    FROM benchmark_executions 
    ORDER BY started_at DESC 
    LIMIT 1;" | xargs)
    
    if [ -z "$EXECUTION_ID" ]; then
        echo "Error: No executions found in database"
        exit 1
    fi
    
    echo "Using latest execution ID from db: $EXECUTION_ID"
else
    echo "Using execution ID from environment: $EXECUTION_ID"
fi

# set default directory if not provided
BENCHMARK_DIR=${BENCHMARK_DIR:-$(pwd)}

cd "$BENCHMARK_DIR"

# run all benchmarks on specific GPU
run_benchmarks_on_gpu() {
    local gpu_index=$1
    local gpu_uuid=$2
    local execution_id=$3
    local benchmark_dir=$4
    
    echo "[GPU $gpu_index] Starting benchmarks on GPU: $gpu_uuid"
    
    # set CUDA_VISIBLE_DEVICES to only show this GPU
    export CUDA_VISIBLE_DEVICES=$gpu_index
    
    # create separate working directory for this GPU to avoid conflicts
    local gpu_work_dir="$benchmark_dir/gpu_$gpu_index"
    mkdir -p "$gpu_work_dir"
    
    # copy source files to GPU-specific directory and compile
    echo "[GPU $gpu_index] Setting up workspace..."
    
    # memory microbenchmarks
    echo "[GPU $gpu_index] ======== Memory Microbenchmarks ================"
    
    # gpu-stream microbenchmark
    echo "[GPU $gpu_index] === GPU Stream Microbenchmark ==="
    mkdir -p "$gpu_work_dir/memory/gpu-stream"
    cp -r memory/gpu-stream/* "$gpu_work_dir/memory/gpu-stream/"
    cd "$gpu_work_dir/memory/gpu-stream"
    make clean && make
    echo "[GPU $gpu_index] Running GPU Stream benchmark and inserting results into database..."
    ./cuda-stream "$execution_id" "$gpu_uuid" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    echo "[GPU $gpu_index] GPU Stream microbenchmark completed!"
    cd "$benchmark_dir"
    
    # cuda-memcpy microbenchmark
    echo "[GPU $gpu_index] === CUDA Memory Copy Microbenchmark ==="
    mkdir -p "$gpu_work_dir/memory/cuda-memcpy"
    cp -r memory/cuda-memcpy/* "$gpu_work_dir/memory/cuda-memcpy/"
    cd "$gpu_work_dir/memory/cuda-memcpy"
    make clean && make
    echo "[GPU $gpu_index] Running CUDA Memory Copy benchmark and inserting results into database..."
    ./cuda-memcpy "$execution_id" "$gpu_uuid" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    echo "[GPU $gpu_index] CUDA Memory Copy microbenchmark completed!"
    cd "$benchmark_dir"
    
    # compute microbenchmarks
    echo "[GPU $gpu_index] ======== Compute Microbenchmarks ==============="
    
    # cuda-matmul microbenchmark
    echo "[GPU $gpu_index] === CUDA Matrix Multiplication Microbenchmark ==="
    mkdir -p "$gpu_work_dir/compute/cuda-matmul"
    cp -r compute/cuda-matmul/* "$gpu_work_dir/compute/cuda-matmul/"
    cd "$gpu_work_dir/compute/cuda-matmul"
    make clean && make
    echo "[GPU $gpu_index] Running CUDA Matrix Multiplication benchmark and inserting results into database..."
    ./cuda-matmul "$execution_id" "$gpu_uuid" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    echo "[GPU $gpu_index] CUDA Matrix Multiplication microbenchmark completed!"
    cd "$benchmark_dir"
    
    # cuda-incore microbenchmark
    echo "[GPU $gpu_index] === CUDA In-Core Compute Microbenchmark ==="
    mkdir -p "$gpu_work_dir/compute/cuda-incore"
    cp -r compute/cuda-incore/* "$gpu_work_dir/compute/cuda-incore/"
    cd "$gpu_work_dir/compute/cuda-incore"
    make clean && make
    echo "[GPU $gpu_index] Running CUDA In-Core benchmark and inserting results into database..."
    ./cuda-incore "$execution_id" "$gpu_uuid" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    echo "[GPU $gpu_index] CUDA In-Core microbenchmark completed!"
    cd "$benchmark_dir"
    
    # cleanup GPU-specific workspace
    rm -rf "$gpu_work_dir"
    
    echo "[GPU $gpu_index] All benchmarks completed on GPU: $gpu_uuid"
}

# export function so it's available to subshells
export -f run_benchmarks_on_gpu

# export environment variables needed by function
export DB_PASSWORD DB_HOST DB_PORT DB_USER DB_NAME

echo "Starting GPU benchmarks collection on ${#GPU_UUIDS[@]} GPU(s)..."
echo "Results will be inserted directly into database"
echo "Running benchmarks in parallel..."

# array to store background process PIDs
declare -a pids

# generate benchmark processes for each GPU in parallel
for i in "${!GPU_UUIDS[@]}"; do
    echo "Launching benchmarks for GPU $i (${GPU_UUIDS[$i]})..."
    run_benchmarks_on_gpu "$i" "${GPU_UUIDS[$i]}" "$EXECUTION_ID" "$BENCHMARK_DIR" &
    pids[$i]=$!
done

echo "All benchmark processes launched. Waiting for completion..."

# wait for all background processes to complete
for i in "${!pids[@]}"; do
    wait ${pids[$i]}
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "GPU $i benchmarks completed successfully!"
    else
        echo "ERROR: GPU $i benchmarks failed with exit code $exit_code"
    fi
done

echo ""
echo "All GPU benchmarks completed!"
echo "Results have been inserted directly into database."
