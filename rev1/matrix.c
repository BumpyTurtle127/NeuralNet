#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

mat_fnn * mats[5];

// Local function to initialize matrix and return pointer to it.
mat_fnn * A(int rows, int cols){
	mat_fnn * ret = (mat_fnn*)malloc(sizeof(mat_fnn));
	ret->rows = rows; ret->cols = cols;
	ret->mat = (double**)malloc(rows*sizeof(double*));
	for(int i = 0; i < rows; i++)
		ret->mat[i] = (double*)malloc(cols*sizeof(double));
	return ret;
}

void initSingle_mat(int rows, int cols, double * a, int pos, mat_fnn ** mat){
	if(rows < 1 || cols < 1) return;
	if(*mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = A(rows, cols);
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			mat[pos]->mat[i][j] = a[j + i*cols];
	return;
}

// Function to initialize matrix with values, and save to pos.
void init_mat(int rows, int cols, double ** a, int pos, mat_fnn ** mat){
	if(rows < 1 || cols < 1) return;
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = A(rows, cols);
	for(int i = 0; i < rows; i++)
		for(int j = 0; j < cols; j++)
			mat[pos]->mat[i][j] = a[i][j];
	return;
}

// Function to initialize identity matrix of size a and save to pos.
void identity_mat(int a, int pos, mat_fnn ** mat){
	if(a < 1) return;
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = A(a, a);	
	for(int i = 0; i < a; i++){
		for(int j = 0; j < a; j++){
			if(i == j) mat[pos]->mat[i][j] = 1;
			else mat[pos]->mat[i][j] = 0;
		}
	} return;
}

void swapRow_mat(int row1, int row2, int a, int pos, mat_fnn ** mat){
	if(mat[a] == NULL) return;
	if(row1 > mat[a]->rows || row2 > mat[a]->rows) return;
	row1--; row2--;
	*mats = A(mat[a]->rows, mat[a]->cols);
	for(int i = 0; i < mat[a]->rows; i++){
		for(int j = 0; j < mat[a]->cols; j++){
			if(i == row1) (*mats)->mat[row2][j] = mat[a]->mat[i][j];
			else if(i == row2) (*mats)->mat[row1][j] = mat[a]->mat[i][j];
			else (*mats)->mat[i][j] = mat[a]->mat[i][j];
		}
	}
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void add_mat(int a, int b, int pos, mat_fnn ** mat){
	if(mat[a] == NULL || mat[b] == NULL) return;
	if((mat[a]->cols != mat[b]->cols) || (mat[a]->rows != mat[b]->rows)) return;
	*mats = A(mat[a]->rows, mat[a]->cols);
	for(int i = 0; i < mat[a]->rows; i++)
		for(int j = 0; j < mat[a]->cols; j++)
			(*mats)->mat[i][j] = mat[a]->mat[i][j] + mat[b]->mat[i][j];
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void sub_mat(int a, int b, int pos, mat_fnn ** mat){
	if(mat[a] == NULL || mat[b] == NULL) return;
	if((mat[a]->cols != mat[b]->cols) || (mat[a]->rows != mat[b]->rows)) return;
	*mats = A(mat[a]->rows, mat[a]->cols);
	for(int i = 0; i < mat[a]->rows; i++)
		for(int j = 0; j < mat[a]->cols; j++)
			(*mats)->mat[i][j] = mat[a]->mat[i][j] - mat[b]->mat[i][j];
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void mult_mat(int a, int b, int pos, mat_fnn ** mat){
	if(mat[a] == NULL || mat[b] == NULL) return;
	if(mat[a]->cols != mat[b]->rows) return;
	*mats = A(mat[a]->rows, mat[b]->cols);
	double temp = 0;
	for(int i = 0; i < mat[a]->rows; i++){
		for(int j = 0; j < mat[b]->cols; j++){
			for(int k = 0; k < mat[a]->cols; k++)
				temp += mat[a]->mat[i][k] * mat[b]->mat[k][j];
			(*mats)->mat[i][j] = temp;
			temp = 0;
		}
	}
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void multConst_mat(double a, int b, int pos, mat_fnn ** mat){
	if(mat[b] == NULL) return;
	*mats = A(mat[b]->rows, mat[b]->cols);
	for(int i = 0; i < mats[b]->rows; i++)
		for(int j = 0; j < mats[b]->cols; j++)
			(*mats)->mat[i][j] = a * mat[b]->mat[i][j];
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

/*
void invert_mat(int a, int pos, mat_fnn ** mat){	
	if(mats[a] == NULL) return;
	if(mats[a]->rows != mats[a]->cols) return;
	
	identity_mat(mats[a]->cols, 0);
	augment(a, 0, 0);
	rref_mat(0, 0);
	mat_fnn temp = **mats;
	split_mat(mats[a]->cols, 1, 0, 0);
	if(B(0) == 1){
		*mats = &temp;
		split_mat(mats[a]->cols, 2, 0, 0);
	} else return;
	if(mats[pos] != NULL) destroy_mat(mats[pos]);
	mats[pos] = *mats;
	return;
}
*/

void transpose_mat(int a, int pos, mat_fnn ** mat){
	if(mat[a] == NULL) return;
	*mats = A(mat[a]->cols, mat[a]->rows);
	for(int i = 0; i < mat[a]->rows; i++)
		for(int j = 0; j < mat[a]->cols; j++)
			(*mats)->mat[j][i] = mat[a]->mat[i][j];
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void augment(int a, int b, int pos, mat_fnn ** mat){
	if(mat[a] == NULL || mat[b] == NULL) return;
	if(mat[a]->rows != mat[b]->rows) return;
	*mats = A(mat[a]->rows, mat[a]->cols + mat[b]->cols);
	int k = mat[a]->cols;
	for(int i = 0; i < mat[a]->rows; i++){
		for(int j = 0; j < mat[a]->cols; j++)
			(*mats)->mat[i][j] = mat[a]->mat[i][j];
		for(int j = 0; j < mats[b]->cols; j++)
			(*mats)->mat[i][j+k] = mat[b]->mat[i][j];
	}
	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

int isZero(int a, mat_fnn ** mat){
	if(mat[a] == NULL) return -1;
	if(mat[a]->mat[0][0] > 1e-2 || mat[a]->mat[0][0] < -1e-2) return 0;
	else return 1;
}

/*
void forwardPhase(int a){
	if(mat[a] == NULL) return;
	int numCycles = 0, maxCycles = mat[a]->rows - 1;
	multConst_mat(1, 0, 101);
	while(isZero(a) == 1 && numCycles < maxCycles){
		multConst_mat(1, 0, 101);
		swapRow_mat(1, numCycles+2, 101, 0);
		numCycles++;
	}
	if(numCycles == maxCycles && isZero(a) == 1){
		
	}
}

void rref_mat(int a, int pos, mat_fnn ** mat){
	if(mat[a] == NULL) return;
	*mats = A(mat[a]->rows, mat[a]->cols);
	double * 
	for(int i = 0; i < mat[a]->rows; i++)
		for(int j = 0; j < mat[a]->cols; j++)
			(*mats)->mat[i][j] = mat[a]->mat[i][j];
	forwardPhase();
}
*/

void split_mat(int numCols, int LorR, int a, int pos, mat_fnn ** mat){
	if(mat[a] == NULL || numCols > mat[a]->cols) return;
	*mats = A(mat[a]->rows, numCols);
	int k = mat[a]->cols - numCols;
	if(LorR == 1){
		for(int i = 0; i < mat[a]->rows; i++)
			for(int j = 0; j < numCols; j++)
				(*mats)->mat[i][j] = mat[a]->mat[i][j];
	} else if(LorR == 2){
		for(int i = 0; i < mat[a]->rows; i++)
			for(int j = 0; j < numCols; j++)
				(*mats)->mat[i][j] = mat[a]->mat[i][j+k];
	} else{
		destroy_mat(*mats);
		return;
	}

	if(mat[pos] != NULL) destroy_mat(mat[pos]);
	mat[pos] = *mats;
	return;
}

void print_mat(int pos, mat_fnn ** mat){
	if(mat[pos] == NULL) return;
	for(int i = 0; i < mat[pos]->rows; i++){
		printf("|* ");
		for(int j = 0; j < mat[pos]->cols; j++){
			printf("%lf ", mat[pos]->mat[i][j]);
		} printf("*|\n");
	} printf("\n");
	return;
}

void destroy_mat(mat_fnn * a){
	if(a == NULL) return;
	for(int i = 0; i < a->rows; i++)
		free(a->mat[i]);
	free(a->mat);
	free(a);
	return;
}

void insertVal_mat(int row, int col, double val, int pos, mat_fnn ** mat){
	if(mat[pos] == NULL) return;
	mat[pos]->mat[row][col] = val;
	return;
}

double getVal_mat(int row, int col, int pos, mat_fnn ** mat){
	if(mat[pos] == NULL) return 0;
	return mat[pos]->mat[row][col];
}
