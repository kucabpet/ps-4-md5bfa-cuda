CC:=nvcc
#PT:=-ccbin=/usr/bin/gcc-4.9
OPT:=-Xcompiler -fPIC -arch=sm_21 -ccbin=/usr/bin/gcc-4.9

main2: cuda2.o main.o
	${CC} $^ -o $@
	export LD_LIBRARY_PATH=/home/edu/cuda2/lib

cuda2.o: cuda2.cu
	${CC} ${OPT}  -c $^  -o $@

main.o: main.cpp
	${CC} ${OPT}  -c $^  -o $@

clean:
	rm *.o main
