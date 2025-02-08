#include <stdio.h>
#include <stdlib.h>

#ifndef NUMINOUTS
#define NUMINOUTS 10
#endif

typedef struct fnnMat{
	double ** mat;
	int rows, cols;
} fnnMat;

typedef struct{
	double ** inputs;
	double ** outputs;
	int inputWidth, outputWidth, height;
} fnnData;

typedef struct{
	fnnMat *** layers;
	fnnMat ** weights;
	int numLayers, *numNodes;
	double err;
} fnnNet;

/* fnn.c */
/* lineNum indexing starts at 0 */
/* Each function returns the loss as MSE */
fnnNet * initNet(int numLayers, int * numNodes);
fnnNet * initNetFromFile(char * filename);
fnnMat * dActivate(fnnMat * vec);
double feedData(fnnData * data, fnnNet * net, int lineNum);
double backprop(fnnData * data, fnnNet * net, int lineNum);
double singleEpoch(fnnData * data, fnnNet * net);
double trainNet(fnnData * data, fnnNet * net, int numEpochs);
void trainNetWithVisual(fnnData * data, fnnNet * net, int numEpochs);
fnnMat * activateVec(fnnMat * vec, char activation);
void printWeightsToFile(fnnNet * net, char * filename);
void printWeights(fnnNet * net);
void destroyNet(fnnNet * a);

/* csv.c */
fnnData * load_fnnData(int numInputs, int numOutputs, char order, char * filename);
int getInputW_fnnData(fnnData * a);
int getOutputW_fnnData(fnnData * a);
int getHeight_fnnData(fnnData * a);
void destroy_fnnData(fnnData * a);

/* rdp.c */
/* Returns array of doubles. First element is number of values */
double * parseString(char * line, int numInOuts);

/* fnnMat.c */
/* All rows and cols start at 0 EXCEPT constructors */
/* All matrix/row operations are first = first <op> second */
double len_fnnMat(fnnMat * a);
void sE(fnnMat ** a, fnnMat ** b); // Memory safe a = b
fnnMat * init_fnnMat(int rows, int cols);
fnnMat * initIdentity_fnnMat(int width);
fnnMat * pop_1d_fnnMat(int rows, int cols, double * a);
fnnMat * pop_2d_fnnMat(int rows, int cols, double ** a);
fnnMat * duplicate(fnnMat * a);

void insertVal_fnnMat(double val, int rows, int cols, fnnMat * a);
double getVal_fnnMat(int rows, int cols, fnnMat * a);

void swapRow_fnnMat(int first, int second, fnnMat * a);
void multRow_fnnMat(double coefficient, int row, fnnMat * a);
void addRow_fnnMat(int first, int second, fnnMat * a);
void subRow_fnnMat(int first, int second, fnnMat * a);

int getRows(fnnMat * a);
int getCols(fnnMat * a);

void rref_fnnMat(fnnMat ** a);
void add_fnnMat(fnnMat ** a, fnnMat ** b);
void sub_fnnMat(fnnMat ** a, fnnMat ** b);
void mult_fnnMat(fnnMat ** a, fnnMat ** b);
void multConst_fnnMat(double coefficient, fnnMat ** a);
void invert_fnnMat(fnnMat ** a);
void transpose_fnnMat(fnnMat ** a);
void augment_fnnMat(fnnMat ** a, fnnMat ** b);
void split_fnnMat(fnnMat ** a, int numCols, int LorR); // 0 for left, anything else for right.
void getCol_fnnMat(fnnMat ** a, int colNum);
void hadamard_fnnMat(fnnMat ** a, fnnMat ** b);

void print_fnnMat(fnnMat * a);
void destroy_fnnMat(fnnMat ** a);
