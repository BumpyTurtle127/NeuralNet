// As a rule of thumb, indexes start at 0, and declarations start at 1.
// For instance, data_fnn a = init_data(5, 2, 100) will create an empty training data struct that is
// 100x10, where the 10 is just 5+5. But getCell_data(1, 1, a) will NOT give you the top leftmost value
// on the input side. getCell_data(0, 0, a) WILL. It's just like int a[10] gives you an array with 10
// values, but a[0] is the first position.
//
// Functions that return pointers will return NULL if there's an error.

#include <stdio.h>

typedef struct mat_fnn{
	double ** mat;
	int rows, cols;
} mat_fnn;

typedef struct data_fnn{
	double ** inputData;
	double ** outputData;
	int inputW, outputW, height;
} data_fnn;

typedef struct net_fnn{
	mat_fnn ** matrices;
	int * weights, ** nodes;
	int numLayers, * numNodes, matNum;
} net_fnn;

// Functions defined in fnn.c
net_fnn * init_net(int numLayers, int * numNodes); 

// Functions defined in csv.c
data_fnn * init_data(int inputW, int outputW, int height); //-------- Returns null if error.
data_fnn * load_data(char * filename); //---------------------------- Returns null if error.
data_fnn * destroy_data(data_fnn * a);
void rmLine_data(int lineNum, data_fnn * a); //---------------------- lines begin at 0.
void addLine_data(int lineNum, double * data, data_fnn * a); //------ NO FAILSAFE.
void getCell_data(int i, int j, data_fnn * a); //-------------------- Start at 0 not 1.
void changeCell_data(int i, int j, double val, data_fnn * a); //----- Start at 0 not 1.

// Functions defined in matrix.c
// Note: pos is 0-100, and location of matrix. 0 is buffer.
void initSingle_mat(int rows, int cols, double * a, int pos, mat_fnn ** mat);
void init_mat(int rows, int cols, double ** a, int pos, mat_fnn ** mat); //--------------- Start at 1 not 0.
void identity_mat(int a, int pos, mat_fnn ** mat);
void swapRow_mat(int row1, int row2, int a, int pos, mat_fnn ** mat);
void add_mat(int a, int b, int pos, mat_fnn ** mat);
void sub_mat(int a, int b, int pos, mat_fnn ** mat);
void mult_mat(int a, int b, int pos, mat_fnn ** mat);
void multConst_mat(double a, int b, int pos, mat_fnn ** mat);
void invert_mat(int a, int pos, mat_fnn ** mat);
void transpose_mat(int a, int pos, mat_fnn ** mat);
void augment(int a, int b, int pos, mat_fnn ** mat);
void rref_mat(int a, int pos, mat_fnn ** mat);
void split_mat(int numCols, int LorR, int a, int pos, mat_fnn ** mat); //---------- 1 for left, 2 for right.
void destroy_mat(mat_fnn * a);
void print_mat(int pos, mat_fnn ** mat);
void insertVal_mat(int row, int col, double val, int pos, mat_fnn ** mat); //-- Start at 0 not 1 for rows and cols.
double getVal_mat(int row, int col, int pos, mat_fnn ** mat); //----------------- Start at 0 not 1 for rows and cols.
