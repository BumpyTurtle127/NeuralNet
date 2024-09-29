
//             BNF Grammar
//--------------------------------------
// <A> := <B>,<B> | <B>\n
// <B> := <S><B> | <C> | <C><S>
// <C> := <D> | <D>.<D>
// <D> := <E> | <E><D>
// <E> := 0 ... 9
// <S> := _ | _<S>
//
#include "fnn.h"

char * b;
int RHS = 0, LHS = 0;
double *retVals, dRHS = 0;

int A();
int B();
int C();
int D();
int E();
int S();

double * parseString(char * line, int numInOuts){
	b = line;
	if(retVals != NULL) free(retVals);
	retVals = (double*)malloc((numInOuts+1)*sizeof(double));
	retVals[0] = 0;
	while(A() == 1 && (*b == ',' || *b == '\n')){	
		dRHS = (double)RHS;
		while(dRHS > 1) dRHS = dRHS / 10;
		retVals[((int)retVals[0])] = (double)LHS + dRHS;
		RHS = 0; LHS = 0; b++;
	}
	return retVals;
}

int S(){
	if(*b == ' '){
		b++;
		S(); return 1;
	} return 0;
}

int E(){
	if(*b >= 48 && *b <= 57){
		RHS = 10*RHS + ((int)(*b) - 48);
		b++;
		return 1;
	} return 0;
}

int D(){
	if(E() == 1){
		D();
		return 1;
	} return 0;
}

int C(){
	if(D() == 1){
		LHS = RHS;
		RHS = 0;
		if(*b == '.'){
			b++;
			if(D() == 1){
				return 1;
			} return 0;
		} return 1;
	} return 0;
}

int B(){
	if(S() == 1){
		if(B() == 1){
			return 1;
		} return 0;
	} else if(C() == 1){
		S(); return 1;
	} return 0;
}

int A(){
	if(B() == 1){
		retVals[0] = retVals[0] + 1;
		return 1;
	} return 0;
}
