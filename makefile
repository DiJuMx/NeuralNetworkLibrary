CC = gcc
AR = ar rcs

CFLAGS = -c -Wall -Wextra
LDFLAGS = -lm

OBJS = neuralNetwork.o


all: libneuralNet.a($(OBJS))

clean:
	rm -rf *.o *.a

libneuralNet.a($(OBJS)) : $(OBJS)
	$(AR) $@ $%
	
%.o: %.c
	$(CC) $(CFLAGS) $< -o $@

.PHONY: all clean
