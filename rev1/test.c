#include <stdio.h>
#include <stdlib.h>
#include "fnn.h"

int main(){
	mat_fnn * Mats[5];

	printf("%p\n", Mats[0]);
	double a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	initSingle_mat(3, 3, a, 1, Mats);
	print_mat(1, Mats);

	double b[] = {1, 1, 1};
	initSingle_mat(1, 3, b, 2, Mats);
	print_mat(2, Mats);

	mult_mat(1, 2, 3, Mats);
	print_mat(3, Mats);

	return 0;
}
