// <A> := <B>, | <B>\n
// <B> := <S><C> | <C>
// <C> := <D> | <D><S>
// <D> := <E> | -<E>
// <E> := <F> | <F>.<F>
// <F> := <G> | <G><F>
// <G> := 0 | 1 | ... | 8 | 9
// <S> := (char)32

#include <stdio.h>
#include <stdlib.h>

char * line;
double dVals[100]
int numDVals;

double ctod(char * a){
	
}

int procLine(char * a, double * ret){
	int len = 0; while(a[len] != '\0') len++;
	line = (char*)malloc(len);
	for(int i = 0; i < len; i++) line[i] = a[i];
	for(int i = 0; i < 100; i++) dVals[i] = 0;
	numDVals = 0; A();
	free(line);
	ret = (double*)malloc(numDVals*sizeof(double));
	for(int i = 0; i < numDVals; i++) ret[i] = dVals[i];
	return numDVals;
}
