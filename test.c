#include <stdio.h>
#include "src/fnn.h"

#define LEARNRATE 0.005

int main(){
	int nL[] = {3, 6, 6, 2};
	fnnData * dataa = load_fnnData(3, 2, "./FA.csv");
	fnnNet * nett = initNet(4, nL);
	for(int i = 0; i < 10000; i++){
		feedData(dataa, nett, 0);
		backprop(dataa, nett, 0);
		feedData(dataa, nett, 1);
		backprop(dataa, nett, 1);
		feedData(dataa, nett, 2);
		backprop(dataa, nett, 2);
		feedData(dataa, nett, 3);
		backprop(dataa, nett, 3);
		feedData(dataa, nett, 4);
		backprop(dataa, nett, 4);
		feedData(dataa, nett, 5);
		backprop(dataa, nett, 5);
		feedData(dataa, nett, 6);
		backprop(dataa, nett, 6);
		feedData(dataa, nett, 7);
		backprop(dataa, nett, 7);
	}
	printf("Error: %lf\n", feedData(dataa, nett, 0));
	printf("Error: %lf\n", feedData(dataa, nett, 1));
	printf("Error: %lf\n", feedData(dataa, nett, 2));
	printf("Error: %lf\n", feedData(dataa, nett, 3));
	printf("Error: %lf\n", feedData(dataa, nett, 4));
	printf("Error: %lf\n", feedData(dataa, nett, 5));
	printf("Error: %lf\n", feedData(dataa, nett, 6));
	printf("Error: %lf\n", feedData(dataa, nett, 7));
	printWeights(nett);
	destroyNet(nett);
	return 0;
}
