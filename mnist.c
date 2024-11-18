#include <stdio.h>
#include <stdlib.h>
#include "src/fnn.h"

int main(){
	int nL[] = {784, 300, 200, 100, 1};
	fnnData * dataa = load_fnnData(784, 1, "./mnist_train.csv");
	fnnNet * nett = initNet(5, nL);
	printf("net initialized.\n");

	for(int i = 0; i < 10000; i++){
		feedData(dataa, nett, i);
		backprop(dataa, nett, i);
	}
	printf("training done.\n");
	

	for(int i = 0; i < 10; i++){
		printf("hello?\n");
		printf("Error: %lf. Predicted vs Real: %lf vs %lf\n", feedData(dataa, nett, i), nett->layers[4][1]->mat[0][0], dataa->outputs[0][0]);
	}
}
