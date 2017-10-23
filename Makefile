CC		= nvcc 
EXEC	= out

SOURCES = \
cuda_mmult_kernels.c \
cuda_mmult.c

OBJS = $(SOURCES:.c=.o)

%.o: %.c
	$(CC) -c -O3 -o $@ $<

all: $(OBJS)
	$(CC) -link -L/usr/local/cuda/lib64/ -O3 $(OBJS) -o $(EXEC)

clean:
	@rm -f $(OBJS) $(EXEC)
