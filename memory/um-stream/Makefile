# Try multiple methods to find CUDA installation
TEMP_NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME := $(shell if [ -x "$(TEMP_NVCC)" ]; then echo $(TEMP_NVCC) | rev | cut -d'/' -f3- | rev; else echo /usr/local/cuda; fi)
NVCC := $(TEMP_NVCC)

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_90 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L$(CUDA_HOME)/lib64 -lcublas
NAME 		:= um-stream
PREFIX		:= .


$(PREFIX)/$(NAME): main.cu Makefile ../../dtime.hpp ../../MeasurementSeries.hpp ../../gpu-error.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./$(NAME)