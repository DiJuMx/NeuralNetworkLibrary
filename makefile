CC = gcc
AR = ar rcs

CFLAGS = -c -Wall -Wextra -g
LDFLAGS = -lm

OBJS = neuralNetwork.o


all: libneuralNet.a($(OBJS))

clean:
	rm -rf *.o *.a

libneuralNet.a($(OBJS)) : $(OBJS)
	$(AR) $@ $%
	
%.o: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@

.PHONY: all clean
