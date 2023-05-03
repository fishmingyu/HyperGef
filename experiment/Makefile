NVCC = $(CUDA_HOME)/bin/nvcc
CC_FLAGS = -Xcompiler "-fPIC" -shared -std=c++14
FIG7 := fig7 
FIG8 := fig8
FIG9 := fig9 
FIG10 := fig10
RM = -rm -rf
OBJ_DIR := objs

OBJS = $(wildcard $(OBJ_DIR)/*.o)

LL = -lcusparse
INC = -I../HyperGsys/include -I$(CUDA_HOME)/include

all : $(FIG7) $(FIG9) $(FIG8) $(FIG10)

$(FIG7): fig7.o
	$(NVCC) $(INC) $(LL) -o $@ $^

$(FIG8): fig8.o
	$(NVCC) $(INC) $(LL) -o $@ $^

$(FIG9): fig9.o
	$(NVCC) $(INC) $(LL) -o $@ $^

$(FIG10): fig10.o
	$(NVCC) $(INC) $(LL) -o $@ $^

%.o: %.cu
	$(NVCC) $(CC_FLAGS) $(INC) -c $< 

clean:
	rm -f *.o $(TARGET)
	$(RM) -rf $(TARGET) $(OBJ_DIR)