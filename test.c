#include <stdio.h>
#include "src/fnn.h"

#define LEARNRATE 0.005

int main(){
	int nL[] = {2, 3, 3, 1};
	fnnData * dataa = load_fnnData(2, 1, 'i', "./sample_training_data/XOR.csv");
	fnnNet * nett = initNet(4, nL);
	if(dataa == NULL) printf("Failed to open file\n");
	printf("Initial Errors: %d\n", dataa->height);
	for(int i = 0; i < dataa->height; i++)
		printf("Error: %lf\n", feedData(dataa, nett, i));
	for(int i = 0; i < 100000; i++){
		for(int j = 0; j < nett->numLayers; j++){
			feedData(dataa, nett, j);
			backprop(dataa, nett, j);
		}
	}
	printf("Final Errors:\n");
	for(int i = 0; i < dataa->height; i++)
		printf("Error: %lf\n", feedData(dataa, nett, i));
	printWeights(nett);
	printWeightsToFile(nett, "TestXorWeights");
	destroyNet(nett);
	return 0;
}
