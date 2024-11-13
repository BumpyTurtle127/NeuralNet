#include <stdio.h>
#include "src/fnn.h"

int main(){
	int nL[] = {784, 300, 200, 100, 1};
	fnnData * dataa = load_fnnData(784, 1, "./mnist_train.csv");
	fnnNet * nett = initNet(5, nL);

	double temp;
	for(int i = 0; i < 60000; i++){
		temp = dataa->inputs[i][0];
		dataa->inputs[i][0] = 0;
		dataa->outputs[i][0] = temp;
	}

	for(int i = 0; i < 100; i++){
		for(int j = 0; j < 100; j++){
			feedData(dataa, nett, i);
			backprop(dataa, nett, i);
		}
	}

	for(int i = 0; i < 10; i++){
		printf("Error: %lf. Predicted vs Real: %lf vs %lf\n", feedData(dataa, nett, i), nett->layers[4][1]->mat[0][0], dataa->outputs[0][0]);
	}
}
