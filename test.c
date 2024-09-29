#include <stdio.h>
#include "fnn.h"

#define LEARNRATE 0.005

int main(){
	int nL[] = {2, 3, 1}; // Try entering numbers more than 4
	fnnData * dataa = load_fnnData(2, 1, "XOR.csv");
	fnnNet * nett = initNet(3, nL);
	printf("%p\n", nett);
	printf("%p %p\n", nett->layers, nett->layers[1]);
	printf("%p %p\n", nett->layers[1][0], nett->layers[1][1]);
	for(int i = 0; i < 4000; i++){
		feedData(dataa, nett, 0);
		backprop(dataa, nett, 0);
		feedData(dataa, nett, 1);
		backprop(dataa, nett, 1);
		feedData(dataa, nett, 2);
		backprop(dataa, nett, 2);
		feedData(dataa, nett, 3);
		backprop(dataa, nett, 3);
	}
	printf("Error: %lf\n", feedData(dataa, nett, 0));
	printf("Error: %lf\n", feedData(dataa, nett, 1));
	printf("Error: %lf\n", feedData(dataa, nett, 2));
	printf("Error: %lf\n", feedData(dataa, nett, 3));
	printWeights(nett);
	destroyNet(nett);
	return 0;
}
