#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "fnn.h"

#ifndef LEARNRATE
#define LEARNRATE 0.01
#endif

fnnNet * initNet(int numLayers, int * numNodes){
	srand(time(NULL));

	/* Allocate the memory*/
	fnnNet * newNet = (fnnNet*)malloc(sizeof(fnnNet));

	/* Assign the variables */
	newNet->numLayers = numLayers;
	newNet->numNodes = (int*)malloc(numLayers*sizeof(int));
	for(int i = 0; i < numLayers; i++) newNet->numNodes[i] = numNodes[i];

	/* Initialize and randomize matrices */
	newNet->layers = (fnnMat***)malloc(numLayers*sizeof(fnnMat**));
	newNet->weights = (fnnMat**)malloc((numLayers-1)*sizeof(fnnMat*));
	for(int i = 0; i < numLayers; i++){
		newNet->layers[i] = (fnnMat**)malloc(2*sizeof(fnnMat*));
		newNet->layers[i][0] = init_fnnMat(numNodes[i], 1);
		newNet->layers[i][1] = init_fnnMat(numNodes[i], 1);
		if(i < numLayers-1){
			newNet->weights[i] = init_fnnMat(numNodes[i+1], numNodes[i]);
			for(int j = 0; j < numNodes[i+1]; j++){
				for(int k = 0; k < numNodes[i]; k++){
					newNet->weights[i]->mat[j][k] = (double)(rand() % 100) / 100;
				}
			}
		}
	}

	/* Return the network struct */
	return newNet;
}

fnnMat * activateVec(fnnMat * vec, char activation){
	/* Only using ReLU for now (testing) */
	fnnMat * ret = init_fnnMat(vec->rows, vec->cols);
	for(int i = 0; i < vec->rows; i++){
		for(int j = 0; j < vec->cols; j++){
			if(vec->mat[i][j] < 0) ret->mat[i][j] = 0;
			else ret->mat[i][j] = vec->mat[i][j];
		}
	} return ret;
}

fnnMat * dActivate(fnnMat * vec){
	/* Only ReLU for now (testing) */
	fnnMat * ret = init_fnnMat(vec->rows, vec->cols);
	for(int i = 0; i < vec->rows; i++){
		for(int j = 0; j < vec->cols; j++){
			if(vec->mat[i][j] <= 0) ret->mat[i][j] = 0;
			else ret->mat[i][j] = 1;
		}
	} return ret;
}

double feedData(fnnData * data, fnnNet * net, int lineNum){
	/* Return if data or net not initialized */
	if(data == NULL || net == NULL) return 0;
	
	/* Copy the numLine'th line of the data into the input nodes of net */
	/* Input nodes are the POST-ACTIVATED nodes of the first layer */
	fnnMat * temp = pop_1d_fnnMat(net->numNodes[0], 1, data->inputs[lineNum]);
	fnnMat * activatedTemp;
	//sE(&(net->layers[0][0]), &(temp));
	sE(&(net->layers[0][1]), &(temp));

	/* Computations */
	for(int i = 0; i < net->numLayers-1; i++){
		temp = duplicate(net->weights[i]);
		mult_fnnMat(&temp, &(net->layers[i][1])); // smth wrong 1
		sE(&(net->layers[i+1][0]), &temp);
		activatedTemp = activateVec(temp, 'c');
		sE(&(net->layers[i+1][1]), &activatedTemp); // smth wrong 1
	}

	/* Compute Error */
	fnnMat * desiredOut = pop_1d_fnnMat(net->numNodes[net->numLayers-1], 1, data->outputs[lineNum]);
	sub_fnnMat(&desiredOut, &activatedTemp);
	double err = len_fnnMat(desiredOut);
	err *= 0.5 * (1/((double)(net->numNodes[net->numLayers-1])));

	/* Destroy temporary matrices */
	if(desiredOut != NULL) destroy_fnnMat(&desiredOut);

	/* Return error */
	net->err = err;
	return err;
}

/* Function to update all remaining matrices */
int recursiveBackprop(fnnNet * net, fnnMat * dEdO, int matNum){
	if(matNum < 0) return 0;
	fnnMat * local_dEdO = duplicate(dEdO);
	fnnMat * tempIden = initIdentity_fnnMat(net->numNodes[matNum+1]);
	fnnMat * dOdA = duplicate(net->weights[matNum+1]);
		transpose_fnnMat(&dOdA);
	fnnMat * dAdO = dActivate(net->layers[matNum+1][1]);
		hadamard_fnnMat(&dAdO, &tempIden);
	fnnMat * dOdW = duplicate(net->layers[matNum][1]);
		transpose_fnnMat(&dOdW);
	
	mult_fnnMat(&dAdO, &dOdA);
	mult_fnnMat(&dAdO, &local_dEdO);
	if(matNum > 0) recursiveBackprop(net, dAdO, matNum-1);
	mult_fnnMat(&dAdO, &dOdW);

	multConst_fnnMat(LEARNRATE, &dAdO);
	add_fnnMat(&(net->weights[matNum]), &dAdO);
	
	destroy_fnnMat(&tempIden);
	destroy_fnnMat(&local_dEdO);
	destroy_fnnMat(&dOdA);
	destroy_fnnMat(&dAdO);
	destroy_fnnMat(&dOdW);
	return 1;
}

double backprop(fnnData * data, fnnNet * net, int lineNum){
	if(data == NULL || net == NULL) return 0;

	/* Look at backprop.pdf to understand alg */
	/* Depth of network is arbitrary */
	/* First setup dE/dA, dA/dO, and dO/dW */
	int matNum = net->numLayers-2;
	fnnMat * tempIden = initIdentity_fnnMat(net->numNodes[matNum+1]);
	fnnMat * dEdA = pop_1d_fnnMat(data->outputWidth, 1, data->outputs[lineNum]);
		sub_fnnMat(&dEdA, &(net->layers[matNum+1][1]));
	fnnMat * dAdO = dActivate(net->layers[matNum+1][1]);
		hadamard_fnnMat(&dAdO, &tempIden);
	fnnMat * dOdW = duplicate(net->layers[matNum][1]);
		transpose_fnnMat(&dOdW);

	/* Then multiply them */
	/* dAdO = dAdO*dEdA*dOdW */
	mult_fnnMat(&dAdO, &dEdA);
	if(matNum > 0) if(recursiveBackprop(net, dAdO, matNum-1) == 0) return 100;
	mult_fnnMat(&dAdO, &dOdW);

	/* Then multiply by the learning rate and update weight matrix */
	/* Adding instead of subtracting because loss is expected - real out */
	/* which creates a negative when differentiating */
	multConst_fnnMat(LEARNRATE, &dAdO);
	add_fnnMat(&(net->weights[net->numLayers-2]), &dAdO);

	destroy_fnnMat(&tempIden);
	destroy_fnnMat(&dAdO);
	destroy_fnnMat(&dOdW);
	destroy_fnnMat(&dEdA);
	return 0; //feedData(data, net, lineNum);
}

void printWeights(fnnNet * net){
	printf("----------------------------------------------------------------\n");
	for(int i = 0; i < net->numLayers-1; i++){
		printf("The weight matrix between layers %d and %d is:\n", i, i+1);
		print_fnnMat(net->weights[i]);
	}
	printf("----------------------------------------------------------------\n\n");
	return;
}

void destroyNet(fnnNet * net){
	for(int i = 0; i < net->numLayers; i++){
		if(i < net->numLayers-1) free(net->weights[i]);
		for(int j = 0; j < 2; j++) free(net->layers[i][j]);
		free(net->layers[i]);
	}
	free(net->numNodes);
	free(net);
	return;
}
