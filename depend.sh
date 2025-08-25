#!/bin/bash
# dependencies for microbenchmarks

echo "Installing dependencies for microbenchmarks..."

# update package lists
apt-get update

# install build essentials
apt-get install -y build-essential

# CUDA toolkit should be pre-installed in container
# If not available, the benchmarks will fail with "nvcc: not found"
# CUPTI should also be available in CUDA installation

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc (CUDA compiler) not found in PATH"
    echo "CUDA toolkit must be installed for these benchmarks to work"
    echo "Common locations: /usr/local/cuda/bin, /opt/cuda/bin"
fi

# Install postgres
apt-get install -y postgresql-client

echo "Dependencies installation completed!"
