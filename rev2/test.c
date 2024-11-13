#include "fnn.h"

int main(){
	double a[] = {1, 2, 3, 4, 5, 6};
	Matrix * A = popMat(2, 3, a);
	printMat(A);

	double b[] = {1, 1, 1};
	Matrix * B = popMat(3, 1, b);
	printMat(B);

	Matrix * temp = B;
	B = multMat(A, B);
	destroyMat(temp);
	printMat(B);
	return 0;
}
