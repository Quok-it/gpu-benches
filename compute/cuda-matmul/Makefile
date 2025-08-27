# Optional host compiler override. Example: make CCBIN=g++-12
CCBIN ?=
CCBIN_FLAG = $(if $(CCBIN),-ccbin $(CCBIN),)

# Try multiple methods to find CUDA installation
TEMP_NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME := $(shell if [ -x "$(TEMP_NVCC)" ]; then echo $(TEMP_NVCC) | rev | cut -d'/' -f3- | rev; else echo /usr/local/cuda; fi)
NVCC := $(TEMP_NVCC)

# internal flags
NVCCFLAGS   := -std=c++14 -O3 -arch=sm_90 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" $(CCBIN_FLAG)
CCFLAGS     := 
LDFLAGS     := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda -lnvidia-ml -lcublas -lcublasLt -lm -lstdc++
NAME 		:= cuda-matmul
PREFIX		:= .
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include

$(PREFIX)/$(NAME): main.cu Makefile
	echo $(CUDA_HOME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

	echo $(CUDA_HOME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./$(NAME)

.PHONY: clean