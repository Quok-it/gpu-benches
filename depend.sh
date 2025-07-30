#!/bin/bash
# Dependencies for GPU benchmarks collection

echo "Installing dependencies for GPU benchmarks..."

# Update package lists
apt-get update

# Install build essentials
apt-get install -y build-essential

# CUDA toolkit should be pre-installed in container
# CUPTI should be available in CUDA installation

echo "Dependencies installation completed"