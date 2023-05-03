NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -Xcompiler "-fPIC" -shared -std=c++14
TARGET := aggr_proto
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)

LL = -lcusparse
INC = -I../include -I$(CUDA_HOME)/include

all : $(TARGET)

$(TARGET): aggr_proto.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(INC) -c $< 

clean:
	rm -f *.o $(TARGET)
	$(RM) -rf $(TARGET) $(OBJ_DIR)