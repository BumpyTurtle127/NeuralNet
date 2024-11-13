#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fnn.h"

Matrix * initMat(int Rows, int Cols){
	Matrix * newMat = malloc(sizeof(Matrix));
	newMat->rows = Rows;
	newMat->cols = Cols;

	int sizee = Rows * Cols;
	newMat->data = (double*)malloc(sizee*sizeof(double));
	for(int i = 0; i < sizee; i++) newMat->data[i] = 0;

	return newMat;
}

Matrix * initIdentity(int width){
	Matrix * newMat = initMat(width, width);
	int sizee = width * width;
	for(int i = 0; i < sizee; i++){
		if((i % (width+1)) == 0)
			newMat->data[i] = 1;
		else
			newMat->data[i] = 0;
	}

	return newMat;
}

double matSqrt(double a){
	int rounds = 0;
	double oldGuess = 5, error = 100, newGuess;
	while(error > 10e-3){
		newGuess = (a + oldGuess*oldGuess) / (2*oldGuess);
		error = fabs(oldGuess - newGuess);
		oldGuess = newGuess;
		rounds++;
	}

	return newGuess;
}

Matrix * popMatSingle(int Rows, int Cols, double * a){
	Matrix * newMat = malloc(sizeof(Matrix));
	newMat->rows = Rows;
	newMat->cols = Cols;
	
	int sizee = Rows * Cols;
	newMat->data = (double*)malloc(sizee*sizeof(double));
	for(int i = 0; i < sizee; i++) newMat->data[i] = a[i];

	return newMat;
}

Matrix * popMat(int Rows, int Cols, double * a){
	/*Matrix * newMat = malloc(sizeof(Matrix));
	newMat->rows = Rows;
	newMat->cols = Cols;

	int sizee = Rows * Cols;
	newMat->data = (double*)malloc(sizee*sizeof(double));
	for(int i = 0; i < Rows; i++){
		for(int j = 0; j < Cols; j++){
			newMat->data[i*Cols + j] = a[i][j];
		}
	}*/

	Matrix * newMat = popMatSingle(Rows, Cols, a);
	return newMat;
}

int insertVal(double value, int i, int j, Matrix * a){
	int Cols = a->cols;
	a->data[((i-1)*Cols + j) - 1] = value;
	if(a->data[((i-1)*Cols + j) - 1] == value) return 1;
	else return 0;
}

double getVal(int i, int j, Matrix * a){
	return a->data[((i-1)*(a->cols) + j) - 1];
}

double * getMatSingle(Matrix * a){
	return a->data;
}

double ** getMat(Matrix * a){
	double ** arry = (double**)malloc((a->rows)*sizeof(double*));
	for(int i = 0; i < a->rows; i++){
		arry[i] = (double*)malloc((a->cols)*sizeof(double));
	}

	for(int i = 0; i < a->rows; i++){
		for(int j = 0; j < a->cols; j++){
			arry[i][j] = a->data[i*(a->cols) + j];
		}
	}

	return arry;
}

int getRows(Matrix * a){return a->rows;}
int getCols(Matrix * a){return a->cols;}

int swapRow(int first, int second, Matrix * a){
	int Cols = a->cols;

	double * buf = (double*)malloc(Cols*sizeof(double));
	for(int i = 0; i < Cols; i++) buf[i] = a->data[(first-1)*Cols + i];
	for(int i = 0; i < Cols; i++) a->data[(first-1)*Cols + i] = a->data[(second-1)*Cols + i];
	for(int i = 0; i < Cols; i++) a->data[(second-1)*Cols + i] = buf[i];

	free(buf);

	return 1;
}

int multRow(double coefficient, int row, Matrix * a){
	for(int i = 0; i < a->cols; i++) a->data[(row-1)*(a->cols) + i] *= coefficient;
	return 1;
}

int addRow(int first, int second, Matrix * a){ // Second + First, and result goes in Second
	int Cols = a->cols;

	for(int i = 0; i < Cols; i++) a->data[(second-1)*Cols + i] += a->data[(first-1)*Cols + i];

	return 1;
}

int subRow(int first, int second, Matrix * a){ // Second - First, and result goes in Second. Subtract row 1 from 2
	int Cols = a->cols;
	for(int i = 0; i < Cols; i++){
		a->data[(second-1)*Cols + i] -= a->data[(first-1)*Cols + i];
	}
	return 1;
}

Matrix * addMat(Matrix * a, Matrix * b){
	if((a->rows != b->rows) || (a->cols || b->cols)) return a;

	Matrix * newMat = initMat(a->rows, a->cols); int sizee = (a->rows)*(a->cols);
	for(int i = 0; i < sizee; i++){
		newMat->data[i] = a->data[i] + b->data[i];
	}

	return newMat;
}

Matrix * subMat(Matrix * a, Matrix * b){
	if((a->rows != b->rows) || (a->cols != b->cols)) return a;

	Matrix * newMat = initMat(a->rows, a->cols); int sizee = (a->rows)*(a->cols);
	for(int i = 0; i < sizee; i++){
		newMat->data[i] = (a->data[i]) - (b->data[i]);
	}

	return newMat;
}

Matrix * duplicate(Matrix * a){
	Matrix * newMat = popMatSingle(a->rows, a->cols, a->data);
	return newMat;
}

void forwardPhase(int i, int j, Matrix * a, int * pivots, int * size){
	int swapNum = i+1;
	while(fabs(getVal(i, j, a)) < 10e-2 && swapNum <= a->rows){
		swapRow(i, swapNum, a);
		swapNum++;
	}

	double numerator = getVal(i, j, a);
	if(fabs(numerator) <= 10e-3) insertVal(0, i, j, a);
	if(fabs(numerator) > 10e-3){
		pivots[*size] = i; (*size)++;
		pivots[*size] = j; (*size)++;

		for(swapNum = i+1; swapNum <= a->rows; swapNum++){
			double denominator = getVal(swapNum, j, a);
			if(fabs(denominator) > 10e-3){
				double multt = numerator / denominator;
				multRow(multt, swapNum, a);
				subRow(i, swapNum, a);
			}
		}

		if(i < a->rows && j < a->cols)
			forwardPhase(i+1, j+1, a, pivots, size);
	} else if(j < a->cols){forwardPhase(i, j+1, a, pivots, size);}
}

void reversePhase(Matrix * a, int * pivots, int size){
	int i, j;

	for(int k = 0; k < size/2; k++){
		i = pivots[size - 2*k - 2];
		j = pivots[size - 2*k - 1];

		for(int index = 1; index <= (i-1); index++){
			double multt = getVal(i - index, j, a) / getVal(i, j, a);
			multRow(multt, i, a);
			subRow(i, i - index, a);
			multt = 1 / multt;
			multRow(multt, i, a);
		}
	}
}

Matrix * ref(Matrix * a){
	int * pivots = (int*)malloc(100*sizeof(int)); int sizee = 0;
	Matrix * newMat = duplicate(a);
	
	forwardPhase(1, 1, newMat, pivots, &sizee);
	for(int i = 0; i < sizee/2; i++){
		double multt = 1 / getVal(pivots[2*i], pivots[2*i + 1], newMat);
		multRow(multt, pivots[2*i], newMat);
	}

	return newMat;
}

Matrix * rref(Matrix * a){
	int * pivots = (int*)malloc(100*sizeof(int)); int sizee = 0;
	Matrix * newMat = duplicate(a);

	forwardPhase(1, 1, newMat, pivots, &sizee);
	for(int i = 0; i < sizee/2; i++){
		double multt = 1 / getVal(pivots[2*i], pivots[2*i + 1], newMat);
		multRow(multt, pivots[2*i], newMat);
	}
	reversePhase(newMat, pivots, sizee);

	return newMat;
}

Matrix * transpose(Matrix * a){
	Matrix * newMat = initMat(a->cols, a->rows);
	int sizee = (a->cols)*(a->rows);
	int Cols = a->cols, Rows = a->rows;
	for(int i = 1; i <= Rows; i++){
		for(int j = 1; j <= Cols; j++){
			newMat->data[(j-1)*Rows + i-1] = a->data[(i-1)*Cols + j-1];
		}
	}

	return newMat;
}

Matrix * augment(Matrix * a, Matrix * b){
	if(a->rows != b->rows) return a;

	int totCols = a->cols + b->cols;
	Matrix * newMat = initMat(a->rows, totCols);
	for(int i = 0; i < a->rows; i++){
		for(int j = 0; j < a->cols; j++){
			newMat->data[i*totCols + j] = a->data[i*(a->cols) + j];
		}
		for(int j = a->cols; j < totCols; j++){
			newMat->data[i*totCols + j] = b->data[i*(b->cols) + j - a->cols];
		}
	}

	return newMat;
}

Matrix * split(Matrix * a, int b, int c){
	Matrix * splitMat = initMat(a->rows, b);

	if(c == 1){
		for(int i = 0; i < a->rows; i++){
			for(int j = 0; j < b; j++){
				splitMat->data[i*b + j] = a->data[i*(a->cols) + j];
			}
		}
	} else if(c == 2){
		for(int i = 0; i < a->rows; i++){
			for(int j = (a->cols - b); j < a->cols; j++){
				splitMat->data[(i+1)*b + j - a->cols] = a->data[i*(a->cols) + j];
			}
		} 
	} else{return a;}

	return splitMat;
}

Matrix * multMat(Matrix * a, Matrix * b){
	if(a->cols != b->rows) return a;
	int additions = a->cols;

	Matrix * mult = initMat(a->rows, b->cols);
	for(int i = 0; i < a->rows; i++){
		for(int j = 0; j < b->cols; j++){
			double temp = 0;
			for(int k = 0; k < additions; k++){
				temp = temp + (a->data[i*additions + k])*(b->data[k*(b->cols) + j]);
			}
			mult->data[i*(b->cols) + j] = temp;
		}
	}

	return mult;
}

Matrix * hadamard(Matrix * a, Matrix * b){
	if((a->cols != b->cols) || (a->rows != b->rows)) return NULL;

	Matrix * ret = initMat(a->rows, a->cols);
	for(int i = 0; i < a->rows; i++){
		for(int j = 0; j < a->cols; j++){
			insertVal(getVal(i, j, a)*getVal(i, j, b), i, j, ret);
		}
	}

	return ret;
}

Matrix * mult(double coefficient, Matrix * a){
	Matrix * newMat = duplicate(a);
	int sizee = (a->rows)*(a->cols);
	for(int i = 0; i < sizee; i++) newMat->data[i] *= coefficient;
	return newMat;
}

Matrix * invertMat(Matrix * a){
	if(a->rows != a->cols) return a;

	Matrix * b = initIdentity(a->cols);
	b = augment(a, b);
	b = rref(b);
	b = split(b, a->cols, 2);

	return b;
}

double dotProd(Matrix * a, Matrix * b){
	int size1 = (a->rows)*(a->cols);
	int size2 = (b->rows)*(b->cols);
	double prod = 0;

	if(size1 == size2){
		for(int i = 0; i < size1; i++){
			prod += (a->data[i])*(b->data[i]);
		}
	}

	return prod;
}

double matNorm(Matrix * a){
	return matSqrt(dotProd(a, a));
}

Matrix * linRegress(double * x, double * y, int length){
	double * secondColl = (double*)malloc(length*sizeof(double));
	for(int i = 0; i < length; i++) secondColl[i] = 1;
	Matrix * secondCol = popMatSingle(length, 1, secondColl);

	Matrix * A = popMatSingle(length, 1, x);
	Matrix * b = popMatSingle(length, 1, y);
	A = augment(A, secondCol);
	Matrix * At = transpose(A);

	A = multMat(At, A);
	b = multMat(At, b);

	A = invertMat(A);

	b = multMat(A, b);

	return b;
}

Matrix ** QR(Matrix * a){
	Matrix ** v = malloc((a->cols)*sizeof(Matrix*));
	for(int i = 0; i < a->cols; i++) v[i] = colVector(i+1, a);

	for(int i = 0; i < a->cols; i++){
		for(int j = 0; j < i; j++){
			v[i] = subMat(v[i], mult(dotProd(v[i], v[j]), v[j]));
		}
		v[i] = mult(1/matNorm(v[i]), v[i]);
	}

	Matrix ** QR = malloc(2*sizeof(Matrix*));
	QR[0] = duplicate(v[0]);
	for(int i = 1; i < (a->cols); i++){
		QR[0] = augment(QR[0], v[i]);
	}

	if(a->cols == a->rows){
		QR[1] = multMat(invertMat(QR[0]), a);
	} else{
		Matrix * temp = duplicate(QR[0]);

	}

	return QR;
}

Matrix * colVector(int j, Matrix * a){
	Matrix * temp = split(split(a, j, 1), 1, 2);
	return temp;
}

Matrix * rowVector(int i, Matrix * a){
	Matrix * temp = transpose(split(split(transpose(a), i, 1), 1, 2));
	return temp;
}

int destroyMat(Matrix * a){
	free(a);
	if(a == NULL) return 1;
	else return 0;
}

void printMat(Matrix * a){
	int Cols = a->cols;

	for(int i = 1; i <= a->rows; i++){
		printf("|* ");
		for(int j = 1; j <= a->cols; j++){
			printf("%lf ", a->data[((i-1)*Cols + j) - 1]);
		}
		printf("*|\n");
	}

	printf("\n");
}
