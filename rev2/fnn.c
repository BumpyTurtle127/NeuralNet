#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

net_fnn * init_net(int numLayers, int * numNodes, char * activations){
	srand(time(NULL));

	net_fnn * ret = (net_fnn*)malloc(sizeof(net_fnn));

	ret->weights = (Matrix**)malloc((numLayers-1)*sizeof(Matrix*));
	for(int i = 0; i < numLayers-1; i++){
		ret->weights[i] = initMat(numNodes[i+1], numNodes[i]);
		for(int j = 0; j < numNodes[i+1], j++)
			for(int k = 0; k < numNodes[i], k++)
				insertVal(((double)(rand()%10))/10, j+1, k+1, ret);
	}

	ret->nodes = (Matrix***)malloc(numLayers*sizeof(Matrix**));
	for(int i = 0; i < numLayers; i++) ret->nodes[i] = (Matrix**)malloc(2*sizeof(Matrix*));

	ret->numLayers = numLayers;
	ret->numNodes = (int*)malloc(numLayers*sizeof(int));
	for(int i = 0; i < numLayers; i++) ret->numNodes[i] = numNodes[i];

	int numA; for(numA = 0; activations[numA] != '\0', i++);
	ret->activations = (char*)malloc(numA);
	for(int i = 0; i < numA; i++) ret->activations[i] = activations[i]; 

	return ret;
}
