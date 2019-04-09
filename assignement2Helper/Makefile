# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

# all: histogram target_name

HPCAssignment2: HPCAssignment2.cu
	$(CC) HPCAssignment2.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) HPCAssignment2.o  $(LDFLAGS) $(IFLAGS) -o HPCAssignment2

# target_name: object_name.cu
	#$(CC) object_name.cu $(LDFLAGS) $(IFLAGS) -c $<
	#$(CC) object_name.o $(LDFLAGS) $(IFLAGS) -o target_name

clean:
	$(RM) HPCAssignment2 *.o *.~ #add target_name
