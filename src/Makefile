# on Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"
UNAME := $(shell uname)
CC = cc 
OPT = -O3 -march=barcelona -ffast-math -fassociative-math -funroll-loops -msse -msse2 -mfpmath=sse 

ifeq ($(UNAME), Darwin)
LDLIBS = -lblas
else
LDLIBS = -lrt
endif


CFLAGS = $(OPT) -Wall -std=gnu99 
LDFLAGS = -Wall
# librt is needed for clock_gettime


targets = benchmark-naive benchmark-blocked benchmark-blas benchmark-tuned search_optimal
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-tuned.o dgemm-blas.o  search_optimal.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-tuned : benchmark.o dgemm-tuned.o
	$(CC) -o $@ $^ $(LDLIBS)
search_optimal: search_optimal.o dgemm-tuned.o
	$(CC) -o $@ $^ $(LDLIBS)
%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
