
include ../common.mk

OPT = -Xcompiler -fPIC -m64

all: libmgpu.a

libmgpu.a: mgpucontext.o mgpuutil.o
	ar -rcs $@ $+

mgpucontext.o: ../src/mgpucontext.cu
		nvcc $(NVCCFLAGS) $(OPT) -o $@ -c $<

mgpuutil.o: ../src/mgpuutil.cpp
		nvcc $(NVCCFLAGS) $(OPT) -o $@ -c $<


clean : 
	rm -f libmgpu.a mgpuutil.o mgpucontext.o