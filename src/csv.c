#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

fnnData * load_fnnData(int numInputs, int numOutputs, char * filename){
	/* Declare necessary variables */
	int dW = (numInputs + numOutputs) + 2; // dW = dataWidth
	int dW12 = 12*dW;
	char * buffer = (char*)malloc(dW12*sizeof(char));
	double * rV; // rV = returnValues
	
	/* Open CSV file */
	FILE * fp = fopen(filename, "r");
	if(fp == NULL) return NULL;

	/* Find height of data */
	int c, height = 0;
	while((c = fgetc(fp)) != EOF)
		if(c == '\n') height++;
	fseek(fp, 0, SEEK_SET);

	/* Initialize the data struct */
	fnnData * newData = (fnnData*)malloc(sizeof(fnnData));
	newData->inputs = (double**)malloc(height*sizeof(double*));
	newData->outputs = (double**)malloc(height*sizeof(double*));
	for(int i = 0; i < height; i++){
		newData->inputs[i] = (double*)malloc(numInputs*sizeof(double));
		newData->outputs[i] = (double*)malloc(numOutputs*sizeof(double));
	} newData->inputWidth = numInputs; newData->outputWidth = numOutputs; newData->height = height;

	/* Read in and parse each line */
	int lineNum = 0;
	while(fgets(buffer, dW12, fp) != NULL && lineNum <= height){
		rV = parseString(buffer, dW);

		/* Save line to struct */
		for(int i = 0; i < numInputs; i++)
			newData->inputs[lineNum][i] = rV[i+1];
		for(int i = 0; i < numOutputs; i++)
			newData->outputs[lineNum][i] = rV[i+numInputs+1];
		lineNum++;
	}

	/* Close the file and return the data struct */
	fclose(fp); return newData;
}

int getInputW_fnnData(fnnData * a){return a->inputWidth;}
int getOutputW_fnnData(fnnData * a){return a->outputWidth;}
int getHeight_fnnData(fnnData * a){return a->height;}

void destroy_fnnData(fnnData * a){
	/* Free all memory allocated for data */
	if(a->inputs != NULL){
		for(int i = 0; i < a->height; i++)
			free(a->inputs[i]);
		free(a->inputs);
	} if(a->outputs != NULL){
		for(int i = 0; i < a->height; i++)
			free(a->outputs[i]);
		free(a->outputs);
	}

	/* Free the data struct and return */
	free(a); return;
}
