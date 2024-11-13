#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

double len_fnnMat(fnnMat * a){
	/* Return 0 if not vector */
	if(a->cols != 1 && a->rows != 1) return 0;

	/* Find sum of squares */
	double total = 0;
	for(int j = 0; j < a->cols; j++)
		for(int i = 0; i < a->rows; i++)
			total += (a->mat[i][j])*(a->mat[i][j]);

	/* Find square root */
	/* Variables are oldGuess error newGuess */
	double oG = 5, er = 100, nG;
	while(er > 10e-3){
		nG = (total + oG*oG) / (2*oG);
		er = oG - nG;
		if(er < 0) er *= -1;
		oG = nG;
	}

	/* Return answer */
	return total; // TEMPORARY
}

void sE(fnnMat ** a, fnnMat ** b){
	if(*a != NULL) destroy_fnnMat(a); // smth wrong 1
	(*a) = (*b);
	return;
}

fnnMat * init_fnnMat(int rows, int cols){
	/* Allocate memory for the matrix */
	if(rows == 0 || cols == 0){
		printf("Rows or cols are 0.\n");
		return NULL;
	} fnnMat * ret = (fnnMat*)malloc(sizeof(fnnMat));

	/* Initialize the rows and columns and set to 0 */
	ret->mat = (double**)malloc(rows*sizeof(double*));
	for(int i = 0; i < rows; i++){
		ret->mat[i] = (double*)malloc(cols*sizeof(double));
		for(int j = 0; j < cols; j++) ret->mat[i][j] = 0;
	}

	/* Set row and col variables to num rows and cols */
	ret->rows = rows; ret->cols = cols;

	/* Return the initialized matrix */
	return ret;	
}

fnnMat * initIdentity_fnnMat(int width){
	/* Allocate memory for the matrix */
	fnnMat * ret = (fnnMat*)malloc(sizeof(fnnMat));

	/* Initialize the rows and columns */
	ret->mat = (double**)malloc(width*sizeof(double*));
	for(int i = 0; i < width; i++){
		ret->mat[i] = (double*)malloc(width*sizeof(double));
		for(int j = 0; j < width; j++){
			if(i == j) ret->mat[i][j] = 1;
			else ret->mat[i][j] = 0;
		}
	}

	/* Set row and col variables to num rows and cols */
	ret->rows = width; ret->cols = width;

	/* Return the initialized matrix */
	return ret;
}

fnnMat * pop_1d_fnnMat(int rows, int cols, double * a){
	/* Allocate memory for the matrix */
	fnnMat * ret = (fnnMat*)malloc(sizeof(fnnMat));

	/* Initialize the rows and columns */
	ret->mat = (double**)malloc(rows*sizeof(double*));
	for(int i = 0; i < rows; i++){
		ret->mat[i] = (double*)malloc(cols*sizeof(double));
		for(int j = 0; j < cols; j++) ret->mat[i][j] = a[i*cols + j];
	}

	/* Set row and col variables to num rows and cols */
	ret->rows = rows; ret->cols = cols;

	/* Return the initialized matrix */
	return ret;	
}

fnnMat * pop_2d_fnnMat(int rows, int cols, double ** a){
	/* Allocate memory for the matrix */
	fnnMat * ret = (fnnMat*)malloc(sizeof(fnnMat));

	/* Initialize the rows and columns */
	ret->mat = (double**)malloc(rows*sizeof(double*));
	for(int i = 0; i < rows; i++){
		ret->mat[i] = (double*)malloc(cols*sizeof(double));
		for(int j = 0; j < cols; j++) ret->mat[i][j] = a[i][j];
	}

	/* Set row and col variables to num rows and cols */
	ret->rows = rows; ret->cols = cols;

	/* Return the initialized matrix */
	return ret;	
}

fnnMat * duplicate(fnnMat * a){
	/* Allocate memory for the matrix */
	fnnMat * ret = (fnnMat*)malloc(sizeof(fnnMat));

	/* Initialize and copy the rows and columns */
	ret->mat = (double**)malloc((a->rows)*sizeof(double*));
	for(int i = 0; i < a->rows; i++){
		ret->mat[i] = (double*)malloc((a->cols)*sizeof(double));
		for(int j = 0; j < a->cols; j++) ret->mat[i][j] = a->mat[i][j];
	}

	/* Set row and col variables to num rows and cols */
	ret->rows = a->rows; ret->cols = a->cols;

	/* Return the initialized matrix */
	return ret;	
}

void insertVal_fnnMat(double val, int rows, int cols, fnnMat * a){
	/* Insert the value it's not that hard */
	a->mat[rows][cols] = val;
	return;
}

double getVal_fnnMat(int rows, int cols, fnnMat * a){
	/* Why is there even a function for this */
	return a->mat[rows][cols];
}

void swapRow_fnnMat(int first, int second, fnnMat * a){
	/* Create temporary variable */
	double temp;

	/* First copy first line into temp */
	/* Then copy second line into first */
	/* Then copy temp into second line */
	for(int i = 0; i < a->cols; i++){
		temp = a->mat[first][i];
		a->mat[first][i] = a->mat[second][i];
		a->mat[second][i] = temp;
	}

	/* Return */
	return;
}

void multRow_fnnMat(double coefficient, int row, fnnMat * a){
	/* Multiply the row and return */
	for(int i = 0; i < a->cols; i++) a->mat[row][i] *= coefficient;
	return;
}

void addRow_fnnMat(int first, int second, fnnMat * a){
	/* Add all elements of 2nd row to 1st and return */
	for(int i = 0; i < a->cols; i++) a->mat[first][i] += a->mat[second][i];
	return;
}

void subRow_fnnMat(int first, int second, fnnMat * a){
	/* Subtract all elements of 2nd row from first row and return */
	for(int i = 0; i < a->cols; i++) a->mat[first][i] -= a->mat[second][i];
	return;	
}

double fnn_abs(double a){if(a < 0) a *= -1; return a;}

void forwardPhase(int i, int j, fnnMat * a, int ** pivots, int * size){
	int swapNum = i+1;

	/* If pivot is 0 swap rows until pivot is found */
	while(fnn_abs(a->mat[i][j]) < 10e-2 && swapNum < a->rows){
		insertVal_fnnMat(0, i, j, a);
		swapRow_fnnMat(i, swapNum, a);
		swapNum++;
	}

	/* If there is a pivot, do the elimination */
	/* Otherwise, go to the next column */
	double numerator = a->mat[i][j];
	if(fnn_abs(numerator) > 10e-2){
		/* Save pivot to array of pivots */
		pivots[*size][0] = i;
		pivots[*size][1] = j;
		(*size)++;

		/* Eliminate everything below pivot and move to next col recursively */
		for(swapNum = i+1; swapNum < a->rows; swapNum++){
			double denominator = a->mat[swapNum][j];
			if(fnn_abs(denominator) > 10e-2){
				double multt = numerator / denominator;
				multRow_fnnMat(multt, swapNum, a);
				subRow_fnnMat(swapNum, i, a); // POSSIBLE BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!
			}
		}

		if(i+1 < a->rows && j+1 < a->cols)
			forwardPhase(i+1, j+1, a, pivots, size);
	} else if(j+1 < a->cols) forwardPhase(i, j+1, a, pivots, size);

	/* Return */
	return;
}

void reversePhase(fnnMat * a, int ** pivots, int size){
	/* Declare necessary variables */
	int i, j;

	/* Go through each column and remove non-pivots */
	for(int k = size - 1; k >= 0; k--){
		/* Obtain coordinate of last pivot */
		i = pivots[k][0];
		j = pivots[k][1];

		/* Eliminate everything above k'th pivot */
		for(int index = 1; index <= i; index++){
			double multt = (a->mat[i-index][j]);
			multRow_fnnMat(multt, i, a);
			subRow_fnnMat(i-index, i, a);
			multt = 1/multt;
			multRow_fnnMat(multt, i, a);
		}
	}

	return;
}

void rref_fnnMat(fnnMat ** a){
	fnnMat * A = *a;

	/* Declare all necessary variables */
	int sizee = 0;
	int ** pivots = (int**)malloc((A->rows)*sizeof(int*));
	for(int i = 0; i < A->rows; i++){
		pivots[i] = (int*)malloc(2*sizeof(int));
	}
	
	fnnMat * b = duplicate(A);

	/* First perform forward-phase elimination */
	/* Then normalize each row */
	/* Then perform reverse-phase elimination */
	forwardPhase(0, 0, b, pivots, &sizee);
	for(int i = 0; i < sizee; i++){
		double multt = 1 / (b->mat[(pivots[i][0])][(pivots[i][1])]);
		multRow_fnnMat(multt, pivots[i][0], b);
	}
       	reversePhase(b, pivots, sizee);

	/* Free the pivot array */
	//for(int i = 0; i < a->rows; i++){
	//	free(pivots[i]);
	//} free(pivots);
	
	/* Go through matrix and fix -0's */
	for(int i = 0; i < b->rows; i++)
		for(int j = 0; j < b->cols; j++)
			if(b->mat[i][j] == 0) b->mat[i][j] = 0;

	/* Destroy matrix a and replace with b */
	destroy_fnnMat(a);
	(*a) = b;

	/* Return */
	return;
}

void add_fnnMat(fnnMat ** a, fnnMat ** b){
	/* Return if matrices don't match */
	if(*a == NULL || *b == NULL) return;
	if((*a)->rows != (*b)->rows || (*a)->cols != (*b)->cols){
		printf("Matrices %p and %p don't match in add_fnnMat(%d %d %d %d).\n", *a, *b, (*a)->rows, (*a)->cols, (*b)->rows, (*b)->cols);
		return;
	} fnnMat * newMat = init_fnnMat((*a)->rows, (*a)->cols);
	
	/* Add all values of b to a */
	for(int i = 0; i < newMat->rows; i++)
		for(int j = 0; j < newMat->cols; j++)
			newMat->mat[i][j] = (*a)->mat[i][j] + (*b)->mat[i][j];

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void sub_fnnMat(fnnMat ** a, fnnMat ** b){
	/* Return if matrices don't match */
	if((*a)->rows != (*b)->rows || (*a)->cols != (*b)->cols){		
		printf("Matrices %p and %p don't match in sub_fnnMat(%d %d %d %d).\n", *a, *b, (*a)->rows, (*a)->cols, (*b)->rows, (*b)->cols);
		return;
	} fnnMat * newMat = init_fnnMat((*a)->rows, (*a)->cols);
	
	/* Add all values of b to a */
	for(int i = 0; i < newMat->rows; i++)
		for(int j = 0; j < newMat->cols; j++)
			newMat->mat[i][j] = (*a)->mat[i][j] - (*b)->mat[i][j];

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void mult_fnnMat(fnnMat ** a, fnnMat ** b){
	fnnMat * A = *a;
	fnnMat * B = *b;

	/* Return if matrices don't match */
	if((*a)->cols != (*b)->rows){
		printf("Matrices %p and %p do not match.\n", *a, *b);
		printf("%d %d %d %d\n", (*a)->rows, (*a)->cols, (*b)->rows, (*b)->cols);
		return;
	}

	/* Make a new matrix */
	fnnMat * newMat = init_fnnMat((*a)->rows, (*b)->cols);
	newMat->rows = (*a)->rows; newMat->cols = (*b)->cols;

	/* Computations */
	double total = 0;
	for(int i = 0; i < (*a)->rows; i++){
		for(int j = 0; j < (*b)->cols; j++){
			for(int k = 0; k < (*a)->cols; k++){
				total += ((*a)->mat[i][k])*((*b)->mat[k][j]);
			} newMat->mat[i][j] = total; total = 0;
		}
	}

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void multConst_fnnMat(double coefficient, fnnMat ** a){
	fnnMat * A = *a;

	/* Just multiply everything one by one */
	for(int i = 0; i < A->rows; i++)
		for(int j = 0; j < A->cols; j++)
			A->mat[i][j] *= coefficient;

	/* Return */
	return;
}

void invert_fnnMat(fnnMat ** a){
	fnnMat * A = *a;

	/* Return if matrix not square */
	if(A->rows != A->cols) return;

	/* Make new matrix and augment identity */
	fnnMat * newMat = duplicate(A);
	fnnMat * idenA = initIdentity_fnnMat(A->cols);
	augment_fnnMat(&newMat, &idenA);

	/* First compute RREF of matrix */
	/* Then split matrix up again */
	rref_fnnMat(&newMat);
	split_fnnMat(&newMat, A->cols, 1);
	
	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Free IdenA */
	free(idenA);

	/* Return */
	return;
}

void transpose_fnnMat(fnnMat ** a){
	fnnMat * A = *a;

	/* Return if matrix is null */
	if(A == NULL) return;

	/* Create new matrix */
	fnnMat * newMat = init_fnnMat(A->cols, A->rows);

	/* Go through matrix and copy values */
	for(int i = 0; i < A->rows; i++)
		for(int j = 0; j < A->cols; j++)
			newMat->mat[j][i] = A->mat[i][j];

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void augment_fnnMat(fnnMat ** a, fnnMat ** b){
	fnnMat * A = *a, * B = *b;

	/* Return if matrices don't match */
	if(A->rows != B->rows) return;

	/* Create new matrix */
	fnnMat * newMat = init_fnnMat(A->rows, A->cols + B->cols);
	newMat->rows = A->rows; newMat->cols = A->cols + B->cols;

	/* Copy over all elements */
	for(int i = 0; i < A->rows; i++){
		for(int j = 0; j < A->cols; j++)
			newMat->mat[i][j] = A->mat[i][j];
		for(int j = 0; j < B->cols; j++)
			newMat->mat[i][j+(A->cols)] = B->mat[i][j];
	}

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void split_fnnMat(fnnMat ** a, int numCols, int LorR){
	fnnMat * A = *a;

	/* Return if a is null */
	if(A == NULL) return;

	/* Make new matrix */
	fnnMat * newMat = init_fnnMat(A->rows, numCols);

	/* Computation */
	if(LorR == 0){
		for(int i = 0; i < A->rows; i++)
			for(int j = 0; j < numCols; j++)
				newMat->mat[i][j] = A->mat[i][j];
	} else{
		for(int i = 0; i < A->rows; i++)
			for(int j = (A->cols)-numCols; j < A->cols; j++)
				newMat->mat[i][j-(A->cols)+numCols] = A->mat[i][j];
	}

	/* Destroy matrix a and replace with newMat */
	destroy_fnnMat(a);
	(*a) = newMat;

	/* Return */
	return;
}

void getCol_fnnMat(fnnMat ** a, int colNum){
	fnnMat * A = *a;

	/* Return if a is null */
	if(A == NULL) return;

	/* Computation */
	int lefttt = colNum - 1;
	int rightt = A->cols - colNum;
	split_fnnMat(a, lefttt, 1);
	split_fnnMat(a, rightt, 0);

	/* Return */
	return;
}

void hadamard_fnnMat(fnnMat ** a, fnnMat ** b){
	fnnMat * A = *a, * B = *b;

	if((*a)->rows == (*b)->rows && (*a)->cols == (*b)->cols){
		/* Multiply all elements of b into a */
		for(int i = 0; i < A->rows; i++)
			for(int j = 0; j < B->cols; j++)
				(*a)->mat[i][j] *= (*b)->mat[i][j];
	} else if((*a)->cols == 1 && (*a)->rows == (*b)->rows){
		fnnMat * newMat = init_fnnMat((*b)->rows, (*b)->cols);
		newMat->rows = (*b)->rows; newMat->cols = (*b)->cols;
		for(int i = 0; i < (*b)->rows; i++)
			for(int j = 0; j < (*b)->cols; j++)
				newMat->mat[i][j] = ((*a)->mat[i][0])*((*b)->mat[i][j]);
		destroy_fnnMat(a);
		(*a) = newMat;
	} else if((*b)->cols == 1 && (*a)->rows == (*b)->rows){
		fnnMat * newMat = init_fnnMat((*a)->rows, (*a)->cols);
		newMat->rows = (*a)->rows; newMat->cols = (*a)->cols;
		for(int i = 0; i < (*a)->rows; i++)
			for(int j = 0; j < (*a)->cols; j++)
				newMat->mat[i][j] = ((*b)->mat[i][1])*((*a)->mat[i][j]);
		destroy_fnnMat(a);
		(*a) = newMat;
	}
	
	/* Return */
	return;
}

int getRows(fnnMat * a){ return a->rows; }
int getCols(fnnMat * a){ return a->cols; }

void print_fnnMat(fnnMat * a){
	/* Very simple just print it out and return */
	for(int i = 0; i < a->rows; i++){
		printf("|* ");
		for(int j = 0; j < a->cols; j++){
			printf("%lf ", a->mat[i][j]);
		}
		printf("*|\n");
	} printf("\n");
	return;
}

void destroy_fnnMat(fnnMat ** a){
	/* Destroy the rows and cols */
	if(*a == NULL) return;
	for(int i = 0; i < (*a)->rows; i++) free((*a)->mat[i]);
	free((*a)->mat);

	/* Destroy the matrix struct */
	free(*a);

	*a = NULL;

	return;
}
