mnist: mnist.c ./src/fnn.c ./src/fnnMat.c ./src/csv.c ./src/rdp.c
	gcc mnist.c ./src/fnn.c ./src/fnnMat.c ./src/csv.c ./src/rdp.c -o mnist -g

clean:
	rm test
