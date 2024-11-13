// As a rule of thumb, indexes start at 0, and declarations start at 1.
// For instance, data_fnn a = init_data(5, 2, 100) will create an empty training data struct that is
// 100x10, where the 10 is just 5+5. But getCell_data(1, 1, a) will NOT give you the top leftmost value
// on the input side. getCell_data(0, 0, a) WILL. It's just like int a[10] gives you an array with 10
// values, but a[0] is the first position.
//
// Functions that return pointers will return NULL if there's an error.

#include <stdio.h>

typedef struct matrix{
	double * data;
	int rows, cols;
} Matrix;

typedef struct data_fnn{
	double ** inputData;
	double ** outputData;
	int inputW, outputW, height;
} data_fnn;

typedef struct net_fnn{
	Matrix ** weights; //--------------------------------------- weights[0] is between input and first layer.
	Matrix *** nodes; //---------------------------------------- nodes[i][j] is i'th layer, and j is pre-activation or post (0 or 1).
	char * activations;
	int numLayers, * numNodes;
} net_fnn;

// Functions defined in fnn.c
net_fnn * init_net(int numLayers, int * numNodes, char * activations);
void activate(net_fnn * net, int layerNum, char actType);
Matrix * dActivate(net_fnn * net, int layerNum, char actType);
void feedData(data_fnn * data, int lineNum, net_fnn * net);
void backProp(int lineNum, net_fnn * net);
void singleEpoch(data_fnn * data, net_fnn * net);
double trainNet(data_fnn * data, net_fnn * net, int numEpochs);
void printNet(net_fnn * net);
void destroy_net(net_fnn * net);

// Functions defined in csv.c
data_fnn * init_data(int inputW, int outputW, int height); //-------- Returns null if error.
data_fnn * load_data(char * filename); //---------------------------- Returns null if error.
data_fnn * destroy_data(data_fnn * a);
void rmLine_data(int lineNum, data_fnn * a); //---------------------- lines begin at 0.
void addLine_data(int lineNum, double * data, data_fnn * a); //------ NO FAILSAFE.
void getCell_data(int i, int j, data_fnn * a); //-------------------- Start at 0 not 1.
void changeCell_data(int i, int j, double val, data_fnn * a); //----- Start at 0 not 1.

// Functions defined in matrix.c
double matSqrt(double a);
Matrix * initMat(int Rows, int Cols);
Matrix * initIdentity(int width);
Matrix * popMatSingle(int Rows, int Cols, double * a);
Matrix * popMat(int Rows, int Cols, double * a);
int insertVal(double value, int i, int j, Matrix * a);
double getVal(int i, int j, Matrix * a);
double * getMatSingle(Matrix * a);
double ** getMat(Matrix * a);
int swapRow(int first, int second, Matrix * a);
int multRow(double coefficient, int row, Matrix * a);
int addRow(int first, int second, Matrix * a);
int subRow(int first, int second, Matrix * a); // In words, "Second = First row subtracted from second."
Matrix * addMat(Matrix * a, Matrix * b);
Matrix * subMat(Matrix * a, Matrix * b);
void forwardPhase(int i, int j, Matrix * a, int * pivots, int * size);
void reversePhase(Matrix * a, int * pivots, int size);
Matrix * duplicate(Matrix * a);
Matrix * ref(Matrix * a);
Matrix * rref(Matrix * a);
Matrix * transpose(Matrix * a);
Matrix * augment(Matrix * a, Matrix * b);
Matrix * split(Matrix * a, int b, int c);
Matrix * multMat(Matrix * a, Matrix * b);
Matrix * hadamard(Matrix * a, Matrix * b);
Matrix * mult(double coefficient, Matrix * a);
Matrix * invertMat(Matrix * a);
double dotProd(Matrix * a, Matrix * b);
double matNorm(Matrix * a);
int getRows(Matrix * a);
int getCols(Matrix * a);
Matrix * linRegress(double * x, double * y, int length);
Matrix ** QR(Matrix * a);
Matrix * colVector(int j, Matrix * a);
Matrix * rowVector(int i, Matrix * a);
int destroyMat(Matrix * a);
void printMat(Matrix * mat);
